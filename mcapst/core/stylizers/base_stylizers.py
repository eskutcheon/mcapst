from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import functools
import os
from pathlib import Path
import torch
# import torchvision.transforms.v2 as TT
# import torchvision.io as IO
# local imports
from ..models.RevResNet import RevResNet
from ..models.CAPVSTNet import CAPVSTNet
from ..models.containers import FeatureContainer, StylizerArgs
# from ..utils.utils import ensure_file_list_format
from ..utils.img_utils import get_scaled_dims, ensure_batch_tensor, downscaling_resize #, iterable_to_tensor
from .stylizer_builder import process_style_sources


# TODO: this whole file really needs to be cleaned up while eliminating redundant code



# TODO: Desperately need to refactor this and comply with the single responsibility principle a lot better
def transform_preprocess(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        # only showing arguments that should be common to all subclasses of BaseStylizer
        cls: BaseStylizer,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor],
        alpha_c: Union[float, None] = None,
        alpha_s: Union[float, Iterable[float]] = None,
        # should include postprocessor back in as an argument after I think over how the stylizers' structure may change
        #postprocessor: Optional[Callable] = None,
        #mask_paths: Union[str, List[str], None] = None,
        **kwargs
    ) -> torch.Tensor:
        #~ in the near future, this will be replaced by a dynamically-constructed preprocessor; the use of a pydantic StylizerParams object is TBD
        # construct the StylizerArgs object from the received arguments
        args = StylizerArgs(
            style_paths=style_paths,
            alpha_c=alpha_c,
            alpha_s=alpha_s,
            #mask_paths=mask_paths,
            **kwargs
        )
        # Safeguards for ensuring proper formatting of `args.style_paths` + converting to batch tensor
        args.style_paths = process_style_sources(args.style_paths, cls.max_size, down_scale=cls.revnet.down_scale)
        # construct the postprocessor if applicable
        postprocessor = args.construct_postprocessor()
        if postprocessor:
            cls.postprocessor = postprocessor
        # fetch supported arguments dynamically from the class
        supported_args = getattr(cls, "supported_args", [])
        if not supported_args:
            raise AttributeError(f"Class {cls.__class__.__name__} must define `supported_args`.")
        filtered_args = args.as_dict(supported_args)
        return func(cls, sample, **filtered_args)
    return wrapper


def initialize_revnet_model(mode: Literal["photo", "art"], device="cuda") -> RevResNet:
    """ Initializes a Reversible Residual Network model based on the specified mode.
        Args:
            mode (str): Mode of the network, either 'photo' or 'art'.
        Returns:
            RevResNet: The initialized reversible network.
    """
    revnet_args = {
        "nBlocks": [10, 10, 10],
        "nStrides": [1, 2, 2],
        "nChannels": [16, 64, 256],
        "in_channel": 3,
        "mult": 4,
        "hidden_dim": 16 if mode == "photo" else 64,
        "sp_steps": 2 if mode == "photo" else 1,
    }
    return RevResNet(**revnet_args).to(device=device)



class BaseStylizer:
    def __init__(self,
                 mode: Literal["photo", "art"], # a class instance can use only use photorealistic or artistic style transfer modes
                 ckpt: str,                     # path to a Reversible Residual Network pre-trained model checkpoint
                 max_size: int,                 # maximum size to restrict both content and style images
                 postprocessor: Callable|None = None,
                 reg_method: str = "ridge",     # regularization method to apply to the output (if any)
                 train_mode: bool = False):
        mode = {"photorealistic": "photo", "artistic": "art"}.get(mode, mode)
        if mode not in ["photo", "art"]:
            raise ValueError(f"ERROR: only 'photo' and 'art' accepted for 'mode' parameter; got {mode}")
        self.mode = mode
        self.max_size = max_size
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ? NOTE: should decide later whether to include the network parameters hardcoded in _set_revnet should be passed from an augment_cfg object (via policy managers)
        # TODO: might want to move revnet to CAP-VSTNet, simplify all inputs, and have stylizers just handle staging, preprocessing, and postprocessing
            # stylizers can be constructed along with each of these components in factories/builders, and revnet checkpointing can be done by functions
            # CAP-VSTNet could be a torch.nn.Module instance and treating it like one while containing the RevResNet could be helpful
            # BUT that could make it more difficult to use inference like a training-time augmentation - need to rethink other ways
                # that we could use it as an augmentation in parallel like we did with the old `StyleTransferDispatcher`
                # - then I have to consider how we can share weights (maybe with torch.nn.DataParallel); sharing weights is the main issue, but wrapping it could solve the issue
        self.revnet = self._set_revnet(mode, ckpt)
        if train_mode:
            self.revnet.train()
        else:
            self.revnet.eval()
        self.feature_aligner = CAPVSTNet(max_size=self.max_size, train_mode=train_mode, reg_method=reg_method)
        #if postprocessor is not None:
        self.postprocessor = postprocessor

    def _set_revnet(self, mode: Literal["photo", "art"], ckpt_path: str = None):
        """ Sets the reversible network based on the specified mode.
            Args:
                mode (str): Mode of the network, either 'photo' or 'art'.
            Returns:
                RevResNet: The initialized reversible network
        """
        revnet = initialize_revnet_model(mode, device=self.device)
        if isinstance(ckpt_path, (str, Path)) and os.path.exists(ckpt_path):
            self._load_revnet_from_ckpt(revnet, ckpt_path)
        return revnet

    def _load_revnet_from_ckpt(self, revnet: RevResNet, ckpt_path: str):
        """ Initializes and loads the reversible network
            Args:
                ckpt_path (str): Path to the pre-trained model checkpoint
            Returns:
                RevResNet: The initialized reversible network
        """
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=self.device)
        revnet.load_state_dict(checkpoint['state_dict'])
        return revnet

    def stylize(self, content_features: FeatureContainer, style_features: FeatureContainer):
        z_cs = self.feature_aligner.transfer(content_features, style_features)
        with torch.no_grad(): # backward pass through reversible network acts as a decoder from feature space to image space
            stylized = self.revnet(z_cs, forward=False)
        if self.postprocessor is not None:
            stylized = self.postprocessor(stylized)
        del z_cs
        return stylized