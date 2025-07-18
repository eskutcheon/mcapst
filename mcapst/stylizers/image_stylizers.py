from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
# local imports
from .base_stylizers import BaseStylizer, transform_preprocess
from ..models.containers import FeatureContainer, StyleWeights
from ..utils.utils import ensure_file_list_format
from ..utils.label_remapping import SegLabelMapper



# TODO: this whole file really needs to be cleaned up while eliminating redundant code


class BaseImageStylizer(BaseStylizer):
    supported_args = ["style_paths", "alpha_c", "alpha_s"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stylize_from_images(
        self,
        content_img: torch.Tensor,
        style_paths: torch.Tensor, #Union[List[str], List[torch.Tensor], torch.Tensor],
        alpha_c: StyleWeights,
        alpha_s: StyleWeights,
        # leaving the segmentation mask arguments here so that base classes can pass them to this method
        cmask: Optional[torch.Tensor] = None,
        smask: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        #style_images = self.process_style_sources(style_paths) #self.get_style_images(style_paths)
        style_images = style_paths
        #print("type and shapes of style images: ", type(style_images), [img.shape for img in style_images])
        with torch.no_grad(): # forward inference of self.revnet acts as the feature encoder
            content_img = self.preprocess(content_img)
            #self._resize(content_img).to(self.device)
            z_c = self.revnet(content_img, forward=True)
            #z_s = [self.revnet(img, forward=True) for img in style_images]
            z_s = self.revnet(style_images, forward=True)
            #print("style features shapes: ", [z.shape for z in z_s])
        content_feat = FeatureContainer(z_c, "content", alpha_c, mask=cmask, max_size=self.max_size)
        style_feat = FeatureContainer(z_s, "style", alpha_s, mask=smask, max_size=self.max_size)
        #print("style features in container: ", [f.shape for f in style_feat.features])
        pastiche = self.stylize(content_feat, style_feat)
        del content_feat, style_feat
        return pastiche

    # def get_style_images(self, style_paths: List[str]) -> List[torch.Tensor]:
    #     """ Loads and preprocesses style images from the provided paths. """
    #     if all(isinstance(p, str) for p in style_paths):
    #         return [self.preprocess(IO.read_image(p, IO.ImageReadMode.RGB).pin_memory()) for p in style_paths]
    #     elif isinstance(style_paths, (list, tuple)) and all(isinstance(p, torch.Tensor) for p in style_paths):
    #         return [self.preprocess(p) for p in style_paths]
    #     elif isinstance(style_paths, torch.Tensor):
    #         return [self.preprocess(style_paths)]
    #     else:
    #         raise ValueError("`style_paths` must be a list of path-like strings or a list of tensors!")

    @transform_preprocess
    def transform(
        self,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str], torch.Tensor, List[torch.Tensor]],
        alpha_c: Union[float, None],
        alpha_s: Union[float, Iterable[float]],
    ):
        """ handle preprocessing where necessary then call stylize, then handle post-processing before returning the augmented sample """
        # TODO: need to use a dedicated function based on native pytorch decorators for this since they handle nested types natively
        content_img = sample if not isinstance(sample, dict) else sample["img"]
        # NOTE: the decorator turns alpha_c and alpha_s into StyleWeights objects
        pastiche: torch.Tensor = self.stylize_from_images(content_img, style_paths, alpha_c = alpha_c, alpha_s = alpha_s)
        if self.postprocessor:
            pastiche = self.postprocessor(pastiche)
        pastiche = pastiche.clamp(0,1)
        # TODO: remove this in favor of expecting only tensors while using intermediate layers to deal with dictionaries - mostly kept like this for mask support anyway
        if isinstance(sample, dict):
            sample["img"] = pastiche
        return pastiche



class MaskedImageStylizer(BaseImageStylizer):
    def __init__(
        self,
        mode: Literal["photo", "art"],
        ckpt: str,
        max_size: int,
        seg_model_ckpt: Optional[str] = None,
        #& just added this argument so it's never set at the moment - add support for this input later
        seg_labels_path: Optional[str] = None,
        postprocessor: Optional[callable] = None,
        reg_method: str = "ridge"
    ):
        """ A stylizer that supports segmentation mask-guided style transfer for images.
            Args:
                mode (Literal["photo", "art"]): Style transfer mode.
                ckpt (str): Path to the reversible residual network checkpoint.
                max_size (int): Maximum size for resizing images.
                seg_model_ckpt (Optional[str]): Path to the semantic segmentation model checkpoint.
                postprocessor (Optional[Callable]): Callable for post-processing stylized outputs.
                reg_method (str): Regularization method.
        """
        raise NotImplementedError("Not tested and confirmed working yet; also needs refactoring for strictly handling style tensors")
        super().__init__(mode, ckpt, max_size, postprocessor, reg_method)
        self.segmentation_model = self._initialize_segmentation_model(seg_model_ckpt)
        self.label_mapper = SegLabelMapper(mapping_name=seg_labels_path, min_ratio=0.01) if seg_labels_path else None
        # TODO: might want to either raise an error if seg_labels_path is None or just set it to a default path
        


    def load_multi_style_masks(self, mask_paths: Union[str, Iterable[str]]):
        mask_paths = ensure_file_list_format(mask_paths)
        masks: List[torch.Tensor] = []
        for idx, path in enumerate(mask_paths):
            masks.append(IO.read_image(path, IO.ImageReadMode.UNCHANGED))
            while len(masks[idx].shape) < 4:
                masks[idx] = masks[idx].unsqueeze(0)
            masks[idx] = masks[idx].to(device=self.device)
            masks[idx] = self._resize(masks[idx])


    def _initialize_segmentation_model(self, ckpt: Optional[str]) -> Optional[torch.nn.Module]:
        """ Load a semantic segmentation model from a checkpoint.
            Args:
                ckpt (Optional[str]): Path to the segmentation model checkpoint.
            Returns:
                Optional[torch.nn.Module]: Loaded model or None if no checkpoint provided.
        """
        if ckpt is not None:
            raise NotImplementedError("Loading segmentation model from checkpoint not yet implemented.")
        # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        # NOTE: expects normalized (with ImageNet mean and std) tensors in shape [N, 3, H, W]
        model = torch.hub.load('pytorch/vision:v0.20.1', 'deeplabv3_resnet101', pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def generate_content_mask(self, content_img: torch.Tensor) -> torch.Tensor:
        """ Generate a content mask using the segmentation model.
            Args:
                content_img (torch.Tensor): Content image tensor of shape [C, H, W].
            Returns:
                torch.Tensor: One-hot encoded content mask of shape [C, H, W].
        """
        seg_mask: Optional[torch.Tensor] = None
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not initialized. Provide a pre-trained checkpoint during instantiation.")
        with torch.no_grad():
            input_tensor = TT.functional.to_dtype(content_img, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output = self.segmentation_model(input_tensor)['out']
            seg_mask = torch.argmax(output, dim=1, keepdim=False)
        # TODO: need to use SegLabelMapper to align the content and style segmentation classes
        #! FIXCHANGE: might not want one-hot labels at all here
        #one_hot_mask = torch.nn.functional.one_hot(seg_mask.squeeze(0), num_classes=output.shape[1])
        #return one_hot_mask.permute(2, 0, 1).bool()
        return seg_mask # should be a label tensor of shape [B, H, W] with values in [0, num_classes-1]

    def _align_mask_labels(self, content_mask: torch.Tensor, style_masks: List[torch.Tensor]) -> torch.Tensor:
        """ Align the segmentation mask labels between content and style images """
        assert self.label_mapper is not None, "Label mapper not initialized. Provide a valid mapping path."
        assert content_mask.ndim in (3, 4), "Content mask must be a single 3D tensor or 4D batch tensor."
        assert all(m.ndim in (3, 4) for m in style_masks), "All style masks must be 3D or 4D tensors."
        content_mask = self.label_mapper(content_mask)


    @transform_preprocess
    def transform(
        self,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str]],
        # TODO: remove all dependence on `use_segmentation` later since segmentation is done in this new subclass (keeping for now until I change the way arguments are passed)
        use_segmentation: bool = False,
        alpha_c: Union[float, None] = None,
        alpha_s: Union[float, Iterable[float]] = None,
        mask_paths: Union[str, List[str], None] = None,
    ) -> torch.Tensor:
        """ Handle preprocessing, call stylize, and post-process the augmented sample. """
        content_img: torch.Tensor               # content image tensor (either passed directly or read from the `sample` dictionary)
        cmask: Optional[torch.Tensor]           # content segmentation mask
        smask: Optional[List[torch.Tensor]]     # style segmentation mask(s) as a list of tensors (to support multiple styles)
        # actual assignments:
        content_img = sample if not isinstance(sample, dict) else sample["img"]
        cmask = sample.get("mask", None) if use_segmentation and isinstance(sample, dict) else None
        if use_segmentation and cmask is None:
            cmask = self.generate_content_mask(content_img)
        # TODO: if we still allow auto-segmentation, we have to support style mask segmentation like the original project
        smask = self.load_multi_style_masks(mask_paths) if use_segmentation and mask_paths else None
        cmask, smask = self._align_mask_labels(cmask, smask) if use_segmentation and cmask is not None and smask is not None else (cmask, smask)
        pastiche = self.stylize_from_images(content_img,
                                            style_paths=style_paths, alpha_c=alpha_c, alpha_s=alpha_s,
                                            cmask=cmask, smask=smask)
        if self.postprocessor:
            pastiche = self.postprocessor(pastiche)
        pastiche = pastiche.clamp(0,1)
        if isinstance(sample, dict):
            sample["img"] = pastiche
        return pastiche