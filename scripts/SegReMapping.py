import numpy as np

class SegReMapping:
    def __init__(self, mapping_name, min_ratio=0.01):
        """ SegReMapping class
            Parameters:
                mapping_name (str): Path to the label mapping file
                min_ratio (float): Minimum ratio threshold for remapping
        """
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio
        self.label_ipt = []

    def _get_label_info(self, seg):
        """ Get unique labels and their indices from the segmentation
            Parameters:
                seg (ndarray): Segmentation array
            Returns:
                tuple: Unique labels and their indices
        """
        return np.unique(seg, return_inverse=True)

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

    def cross_remapping(self, cont_seg, styl_seg):
        """ Perform cross remapping between content and style segmentations
            Parameters:
            cont_seg (ndarray): Content segmentation array
            styl_seg (ndarray): Style segmentation array
            Returns:
            ndarray: New content segmentation with remapped labels
        """
        cont_labels, cont_indices = self._get_label_info(cont_seg)
        style_labels = np.unique(styl_seg)
        cont_set_diff = set(cont_labels) - set(style_labels) - set(self.label_ipt)
        new_cont_labels = self._update_labels(cont_labels, cont_set_diff, style_labels)
        new_cont_seg = new_cont_labels[cont_indices].reshape(cont_seg.shape)
        return new_cont_seg

    def styl_merge(self, cont_seg, styl_seg):
        """ Merge style segmentation into content segmentation
            Parameters:
                cont_seg (ndarray): Content segmentation array
                styl_seg (ndarray): Style segmentation array
            Returns:
                ndarray: New style segmentation with merged labels
        """
        cont_label_info = np.unique(cont_seg)
        style_label_info = np.unique(styl_seg)
        styl_set_diff = set(style_label_info) - set(cont_label_info) - set(self.label_ipt)
        valid_styl_set = set(style_label_info) - styl_set_diff
        new_style_label_info = self._update_labels(style_label_info, styl_set_diff, valid_styl_set)
        return self._remap_labels(styl_seg, style_label_info, new_style_label_info)

    def self_remapping(self, seg):
        """ Perform self-remapping on the segmentation based on label ratios
            Parameters:
                seg (ndarray): Segmentation array
            Returns:
                ndarray: Segmentation array with self-remapped labels
        """
        label_info, ratio_info = self._calculate_ratios(seg)
        new_label_info = label_info.copy()
        for i, (current_label, ratio) in enumerate(zip(label_info, ratio_info)):
            if ratio < self.min_ratio:
                for new_label in self.label_mapping[:, current_label]:
                    if new_label in label_info:
                        index = np.where(label_info == new_label)[0][0]
                        if ratio_info[index] >= self.min_ratio:
                            new_label_info[i] = new_label
                            break
        return self._remap_labels(seg, label_info, new_label_info)