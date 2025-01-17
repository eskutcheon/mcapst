import torch
import sys
import torch.jit
import torchvision.transforms.v2 as TT
from typing import Dict, List, Literal, Union, Iterable, Callable, Tuple
# local imports
from .containers import FeatureContainer, preprocess_and_postprocess
from .cWCT import cWCT




class CAPVSTNet(object):
    def __init__(self, eps:float = 2e-5, max_size:int = 1280, use_double:bool = True, train_mode:bool = False, reg_method: str = "ridge"):
        """
            :param eps: Small constant for numerical stability in Cholesky decomposition - passed to cWCT constructor
            :param use_double: use double precision for calculations if True
            :param train_mode: training mode flag - passed to cWCT constructor to halt training if torch.linalg.cholesky fails
            * NOTE: GPT-generated docstring
        """
        self.use_double = use_double
        self.train_mode = train_mode # keeping this just in case I may need it for more error checking elsewhere
        self.max_size = max_size
        self.cwct = cWCT(train_mode=train_mode, eps=eps, reg_method=reg_method)


    def transfer(self, cont_feat: FeatureContainer, style_feat: FeatureContainer):
        """ Transfer style from styl_feat to cont_feat, optionally using segmentation masks.
            :param cont_feat: wrapped content features [B, N, H_c, W_c] - in practice, typically B=1
            :param styl_feat: wrapped style features [B, N, H_s, W_s]
            :return: Transferred features [B, N, H_c, W_c]
            * NOTE: GPT-generated docstring
        """
        feature_dict = {
            "content": cont_feat,
            "style": style_feat,
            # ? NOTE: could replace the class member `max_size` and just copy it from the content FeatureContainer instance
            "target": FeatureContainer(cont_feat.feat, "target", alpha=None, mask=None, use_double=cont_feat.use_double, max_size=self.max_size)
        }
        # TODO: will be changing this logic later since I don't see why segmentation guidance and multi-style interpolation should be mutually exclusive
            # may have to do style interpolation then join the style masks together by their intersection
            # will probably be making a new function for that, but I need to make the functions that I have as atomic as possible to just call them in this new function
        # TODO: may create a small factory function that does staging here later - no need for using the arguments this way
        if cont_feat.alpha[0] != 0.0 or style_feat.batch_size != 1:
            return self._interpolation(feature_dict)
        elif cont_feat.mask is not None and style_feat.mask is not None:
            return self._transfer_seg(feature_dict)
        else:
            return self._transfer(feature_dict)


    #@torch.jit.script_method
    @preprocess_and_postprocess
    def _transfer(self, feature_dict: Dict[str, FeatureContainer]):
        """ Perform style transfer without segmentation masks.
            :param cont_feat: Content features [B, N, H_c, W_c]
            :param styl_feat: Style features [B, N, H_s, W_s]
            :return: Transferred features [B, N, H_c, W_c]
            * NOTE: GPT-generated docstring
        """
        # whitening and coloring transforms
        whiten_fea = self.cwct.whitening(feature_dict["content"].feat)
        feature_dict["target"].feat = self.cwct.coloring(whiten_fea, feature_dict["style"].feat)
        # ? NOTE: returning a FeatureContainer object for the decorator to apply post-processing
        return feature_dict["target"]


    # taking the place of compute_label_info plus some other stuff from the calling function
    def _get_masked_target_features(self, content_feat: FeatureContainer, style_feat: FeatureContainer, label_set):
        # using this since lambda I'm thinking about removing the FeatureContainer.get_mask_indices function; though encapsulation for its own sake isn't a bad idea
        #get_channel_indices = lambda onehot_mask, label_idx: torch.nonzero(onehot_mask[label_idx])
        is_valid = lambda a, b: 10 < a < 100*b and 10 < b < 100*a
        """ NOTE on function above
            seems that 10 and 100 were arbitrarily chosen to check that the number of nonzero elements (i.e., pixels with the given label below) was greater than 10
                for both content and style masks and that the proportion of that pixel isn't over 100x greater in one than the other
            - ? NOTE: I have my suspicions that the authors did this last minute after encountering an edge case that resulted in crappy results
        """
        get_num_true = lambda mask: int(torch.count_nonzero(mask).cpu().numpy())
        target_feature = content_feat.feat.clone()    # [B, N, H_c*W_c]
        # ~ IDEA: might be easier to iterate over the batch dimension (less common occurrence anyway) and try to do all labels at once
        # From here on, assume that content_feat's batch dimension will only ever be 1 - need to edit this within all comments

        for label in label_set:
            #content_indices: torch.Tensor = torch.nonzero(content_feat.mask.squeeze(0))    # of shape [num_nonzero, [1,C,H*W]]
            #style_indices: torch.Tensor = torch.nonzero(style_feat.mask)        # of shape [num_nonzero, [B,C,H*W]]
            cmask = content_feat.mask[0, label]
            smask = style_feat.mask[0, label]
            if not is_valid(get_num_true(cmask), get_num_true(smask)):
                continue
            # ? NOTE: previously used `torch.index_select` like `torch.index_select(content_feat, -1, idx_c)` where idx_c = torch.nonzero(cmask)
            masked_content = content_feat.feat[:, :, cmask]
            masked_style = style_feat.feat[:, :, smask]
            # apply WCT to masked features
            feat_whitened = self.cwct.whitening(masked_content)
            feat_colored = self.cwct.coloring(feat_whitened, masked_style)
            temp = target_feature.transpose(-1, -2) # [B, H*W, N]
            temp[:, cmask] = feat_colored.transpose(-1, -2)
            # update temp tensor with colored features, copying at indices given by idx_c in last two dimensions
            #temp.index_copy_(-2, idx_c, feat_colored.transpose(-1, -2))  # [B, H*W, N]
            target_feature = temp.transpose(-1, -2) # [B, N, H*W]
        return target_feature
        # ~ IDEA: could permute the dimensions of the one-hot tensor so that the label channel dim is first and batches are second, in order to iterate over labels and do batches concurrently


    @preprocess_and_postprocess
    def _transfer_seg(self, feature_dict: Dict[str, FeatureContainer]):
        """ Perform style transfer using segmentation masks.
            :param cont_feat: Content features [B, N, H_c, W_c]
            :param styl_feat: Style features [B, N, H_s, W_s]
            :param cmask: Content mask [B, _, _]
            :param smask: Style mask [B, _, _]
            :return: Transferred features [B, N, H_c, W_c]
            * NOTE: GPT-generated docstring
        """
        # doesn't even make sense for any application unless I add mixup-type augmentations into this
        assert feature_dict["content"].batch_size == 1, f"ERROR: batch dimension B of content mask can only be 1; got B={feature_dict['content'].batch_size}"
        if feature_dict["style"].batch_size > 1:
            # TODO: figure how I wanna use the interpolation functions with this later - may need to refactor that function to handle lists of tensors
            raise NotImplementedError # later replace this with a call to a new function instead of self._get_masked_target_features
        # ? NOTE: iteration over batch dim seems pretty unnecessary at the moment since only single style images are given to this function
        # ? NOTE: line below would be all that would have to change if I converted all FeatureContainer masks to one-hot tensors
        #label_set = torch.cat((feature_dict["content"].mask.unique(), feature_dict["style"].mask.unique())).unique() # union of features' mask label values
        label_set = torch.arange(0, feature_dict["content"].mask.shape[1])
        # sort labels to ensure consistency - half sure that torch.unique() and torch.arange() both do this by default, but it's a just-in-case thing
        label_set, _ = label_set.sort(stable=True)
        feature_dict["target"].feat = self._get_masked_target_features(feature_dict["content"], feature_dict["style"], label_set)
        return feature_dict["target"] # [B, N, H_c, W_c]

    def _interpolate_style(self, style_feat_batch: torch.Tensor, alpha_s: List[float], feat_shape: Tuple[int, int]):
        """ Interpolate between multiple style features.
            :param style_feat_batch: Tensor of style features [B_s, N, H_s*W_s]
            :param alpha_s: List of interpolation weights
            :param feat_shape: Shape of the content feature [B_c, N]
            :return: Interpolated Cholesky factors and means [B_c, N, N] and [B_c, N]
        """
        B_c, N = feat_shape
        B_s = style_feat_batch.shape[0]
        conversion_dtype = torch.float32 if not self.use_double else torch.float64
        # Initialize mixed Cholesky factors and means (interpolate between all style inputs)
        alpha_s_tensor = torch.tensor(alpha_s, device=style_feat_batch.device, dtype=conversion_dtype).view(-1, 1)
        mix_Ls = torch.zeros((B_c, N, N), device=style_feat_batch.device, dtype=conversion_dtype)    # [B_c, N, N]
        mix_s_mean = torch.zeros((B_c, N), device=style_feat_batch.device, dtype=conversion_dtype)   # [B_c, N]
        # Get style feature covariances and decompositions (mean, covariance matrix, and Cholesky factor)
        s_mean, _, Ls = self.cwct.get_feature_covariance_and_decomp(style_feat_batch, invert=False, update_mean=False) # [B_s, N, N] and [B_s, N]
        # Vectorized interpolation by broadcasting alpha_s across batches
        alpha_broadcasted_L = alpha_s_tensor.view(B_s, 1, 1).expand(B_s, N, N)  # Expand alphas for matrix multiplication (for Ls)
        alpha_broadcasted_mean = alpha_s_tensor.view(B_s, 1).expand(B_s, N)     # Expand alphas for mean vector
        # Perform weighted sum over the batch of styles
        mix_Ls += torch.sum(alpha_broadcasted_L * Ls.unsqueeze(0), dim=1)  # [B_c, N, N]
        mix_s_mean += torch.sum(alpha_broadcasted_mean * s_mean.unsqueeze(0), dim=1)  # [B_c, N]
        return mix_Ls, mix_s_mean

    #@torch.jit.script_method
    @preprocess_and_postprocess
    def _interpolation(self, feature_dict: Dict[str, FeatureContainer]):
        """ Perform style interpolation between content and multiple styles.
            :param alpha_c: Weight for content interpolation
            :return: Interpolated features [B_c, N, H_c, W_c]
            * NOTE: GPT-generated docstring
        """
        content_feat = feature_dict["content"].feat
        B_c, N, _ = content_feat.shape  # Unpack content tensor shape [B_c, N, H_c*W_c]
        alpha_c = feature_dict["content"].alpha[0]
        alpha_s = feature_dict["style"].alpha.weights
        # get Cholesky decomposition for content features
        c_mean, _, Lc = self.cwct.get_feature_covariance_and_decomp(content_feat, invert=False, update_mean=False)  # [B_c, N] and [B_c, N, N]
        Lc_inv = torch.inverse(Lc)
        # get whitened content features (not using the whitening function since I didn't wanna pass all variables each time)
        whiten_c = torch.bmm(Lc_inv, content_feat) # [B_c, N, H_c*W_c]
        # First interpolate between style_A, style_B, style_C, ...
        #mix_Ls, mix_s_mean = self._interpolate_style(feature_dict["style"].feat, alpha_s, tuple(c_mean.shape[:2])) # [B_c, N, N] and [B_c, N]
        mix_Ls, mix_s_mean = self._interpolate_style(feature_dict["style"].feat, alpha_s, (B_c, N)) # [B_c, N, N] and [B_c, N]
        # Second interpolate between content and style_mix
        if alpha_c != 0.0:
            mix_Ls = mix_Ls*(1 - alpha_c) + Lc*alpha_c
            mix_s_mean = mix_s_mean*(1 - alpha_c) + c_mean*alpha_c
        # color the whitened content features
        color_fea = torch.bmm(mix_Ls, whiten_c)                                                   # [B_c, N, H_c*W_c]
        feature_dict["target"].feat = color_fea + mix_s_mean.unsqueeze(-1).expand_as(color_fea)   # [B_c, N, H_c*W_c]
        return feature_dict["target"] # [1, N, H_c, W_c]