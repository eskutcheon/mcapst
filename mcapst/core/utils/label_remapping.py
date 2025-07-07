import torch
import numpy as np
from pathlib import Path
from typing import Union


# TODO: need to update the label mapper to deal with Pytorch tensors natively - for now, I'll just accept the overhead of conversion
def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """ Convert a PyTorch tensor to a NumPy array to work within the old segmentation label remapping code below """
    if isinstance(tensor, torch.Tensor):
        permutation = (1, 2, 0) if tensor.ndim == 3 else (0, 2, 3, 1) if tensor.ndim == 4 else None
        if permutation is None:
            raise ValueError(f"Unsupported number of tensor dimensions: {tensor.ndim}. Expected 3 or 4 dimensions.")
        tensor = tensor.permute(*permutation).cpu().numpy()
    return tensor

def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """ Convert a NumPy array to a PyTorch tensor """
    if isinstance(array, np.ndarray):
        if array.ndim in (3, 4):
            permutation = (2, 0, 1) if array.ndim == 3 else (0, 3, 1, 2)
            array = torch.from_numpy(array).permute(*permutation)  # HWC to CHW
        else:
            raise ValueError(f"Unsupported number of array dimensions: {array.ndim}. Expected 3 or 4 dimensions.")
    return array



class SegLabelMapper:
    def __init__(self, mapping: Union[str, Path, np.ndarray], min_ratio: float = 0.005):
        """ SegLabelMapper class
            Parameters:
                mapping: Path to a .npy file or a NumPy 2D array of shape (M, C),
                    where each column corresponds to an original label and each row a candidate mapping.
                min_ratio: Minimum fraction of pixels a class must occupy to be considered valid
        """
        # think we're assuming this was meant to be a 2D array where each column corresponded to a label and each row to a new label mapping
        # load and validate the mapping array
        if isinstance(mapping, (str, Path)):
            self.label_mapping = np.load(mapping)
        elif isinstance(mapping, np.ndarray):
            self.label_mapping = mapping
        else:
            raise TypeError("mapping must be a file path or a NumPy array")
        if self.label_mapping.ndim != 2:
            raise ValueError("label_mapping must be a 2D array of shape (M, C)")
        self.min_ratio = min_ratio  # eliminate noisy classes with the "min_ratio" threshold
        #? NOTE: there was an error in the original code where the old `self.label_ipt` was never initialized
            #? but still used in `cross_remapping` and `styl_merge` methods
        # labels provided by user - never remapped
        self.label_inputs: set[int] = set()


    def _update_labels(self, label_info, target_labels, valid_set):
        """ Update labels based on the target labels and valid set
            Parameters:
                label_info (ndarray): Array of label information
                target_labels (set): Set of target labels to update
                valid_set (set): Set of valid labels
            Returns:
                ndarray: Updated label information
        """
        new_label_info = label_info.copy()
        for s in target_labels:
            label_index = np.where(label_info == s)[0][0]
            for new_label in self.label_mapping[:, s]:
                if new_label in valid_set:
                    new_label_info[label_index] = new_label
                    break
        return new_label_info

    def _remap_labels(self, seg, label_info, new_label_info):
        """ Remap labels in the segmentation based on new label information
            Parameters:
                seg (ndarray): Segmentation array
                label_info (ndarray): Original label information
                new_label_info (ndarray): New label information
            Returns:
                ndarray: Segmentation array with remapped labels
        """
        new_seg = np.copy(seg)
        for current_label, new_label in zip(label_info, new_label_info):
            new_seg[seg == current_label] = new_label
        return new_seg

    def cross_remapping(
        self,
        content_mask: Union[np.ndarray, torch.Tensor],
        style_mask: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """ Perform cross remapping between content and style segmentations
            Parameters:
                content_mask: Content segmentation mask (H, W) as NumPy array or torch.Tensor
                style_mask: Style segmentation mask (H, W) as NumPy array or torch.Tensor
            Returns:
                torch.Tensor: mask with labels remapped to nearest valid style labels
        """
        # get unique labels and their indices from the segmentation mask
        content_mask_np = torch_to_numpy(content_mask)
        style_mask_np = torch_to_numpy(style_mask)
        cont_labels, cont_indices = np.unique(content_mask_np, return_inverse=True)
        style_labels = set(np.unique(style_mask_np))
        #cont_set_diff = set(cont_labels) - set(style_labels) - set(self.label_inputs)
        # new_cont_labels = self._update_labels(cont_labels, cont_set_diff, style_labels)
        # new_cont_seg = new_cont_labels[cont_indices].reshape(content_mask.shape)
        # return new_cont_seg
        # lookup table for remapping
        lookup = {lbl: lbl for lbl in cont_labels}
        # determine labels to remap
        targets = [lbl for lbl in cont_labels if not (lbl in style_labels or lbl in self.label_inputs)]
        for lbl in targets:
            for candidate in self.label_mapping[:, lbl]:
                if candidate in style_labels:
                    lookup[lbl] = int(candidate)
                    break
        # vectorized application of the lookup table
        remapped_flat = np.array([lookup[lbl] for lbl in cont_labels], dtype=cont_labels.dtype)[cont_indices]
        remapped = remapped_flat.reshape(content_mask_np.shape)
        # TODO: may just make a decorator for this in the short term if I don't refactor for PyTorch tensors
        if isinstance(content_mask, torch.Tensor):
            remapped = numpy_to_torch(remapped).to(dtype=content_mask.dtype, device=content_mask.device)
        return remapped

    # #! determine if this is even necessary to keep as the original authors commented it out on their final scripts
    # def style_merge(self,
    #     cont_seg: Union[np.ndarray, torch.Tensor],
    #     styl_seg: Union[np.ndarray, torch.Tensor]
    # ) -> Union[np.ndarray, torch.Tensor]:
    #     """ Merge style segmentation into content segmentation by remapping any style-only labels
    #         Parameters:
    #             cont_seg: Content segmentation array
    #             styl_seg: Style segmentation array
    #         Returns:
    #             ndarray: New style segmentation mask with merged labels remapped to the nearest valid content labels
    #     """
    #     cont_label_info = np.unique(cont_seg)
    #     style_label_info = np.unique(styl_seg)
    #     styl_set_diff = set(style_label_info) - set(cont_label_info) - set(self.label_inputs)
    #     valid_styl_set = set(style_label_info) - styl_set_diff
    #     new_style_label_info = self._update_labels(style_label_info, styl_set_diff, valid_styl_set)
    #     return self._remap_labels(styl_seg, style_label_info, new_style_label_info)


    def _calculate_ratios(self, seg):
        """ Calculate the ratio of each label in the segmentation
            Parameters:
                seg (ndarray): Segmentation array
            Returns:
                tuple: Label information and their corresponding ratios
        """
        n_pixels = seg.size
        label_info, counts = np.unique(seg, return_counts=True)
        ratio_info = counts / n_pixels
        return label_info, ratio_info

    def self_remapping(self, seg: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """ Reassign mask labels based on label ratios and remove noisy minority classes by mapping them to larger ones
            Parameters:
                seg: Segmentation mask array
            Returns:
                segmentation mask with self-remapped labels
        """
        seg_np = torch_to_numpy(seg)
        labels, ratios = self._calculate_ratios(seg_np)
        # new lookup table for remapping labels
        lookup = {lbl: lbl for lbl in labels}
        # new_labels = labels.copy()
        # for i, (current_label, ratio) in enumerate(zip(labels, ratios)):
        #     if ratio < self.min_ratio:
        #         for new_label in self.label_mapping[:, current_label]:
        #             if new_label in labels:
        #                 index = np.where(labels == new_label)[0][0]
        #                 if ratios[index] >= self.min_ratio:
        #                     new_labels[i] = new_label
        #                     break
        # return self._remap_labels(seg, labels, new_labels)
        # find small labeled regions and remap them to larger ones
        outliers = [lbl for lbl, r in zip(labels, ratios) if r < self.min_ratio]
        valid_labels = set(labels) - set(outliers) # set difference to get valid labels
        for lbl in outliers:
            for candidate in self.label_mapping[:, lbl]:
                if candidate in valid_labels:
                    lookup[lbl] = int(candidate)
                    break
        remapped_flat = np.array([lookup[lbl] for lbl in seg_np.flatten()], dtype=seg_np.dtype)
        remapped = remapped_flat.reshape(seg_np.shape)
        if isinstance(seg, torch.Tensor):
            remapped = numpy_to_torch(remapped).to(dtype=seg.dtype, device=seg.device)
        return remapped


    # TODO: add factory method to instantiate SegLabelMapper from input labels (content and styles)