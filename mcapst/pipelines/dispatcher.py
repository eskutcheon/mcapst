from typing import Union, Tuple, List  #Literal, Any, Dict, Type, Callable, Optional
import os
import torch
from threading import Semaphore, Lock
from queue import Queue

from mcapst.stylizers.image_stylizers import BaseImageStylizer
# TODO: need to implement this for the remaining stylizer subclasses, including for video


class StyleTransferDispatcher:
    def __init__(self, transfer_type, ckpt_dir, max_size, num_copies: Union[int, List[int], Tuple[int]] = 2, use_cuda_streams=True):
        """
        Initializes multiple style managers for parallel or asynchronous style transfer.
        Args:
            transfer_type (list): Type of style transfer ('art', 'photo', etc.).
            ckpt_dir (str): Path to the checkpoint directory for pretrained models.
            max_size (int): Max size for style transfer processing.
            num_copies (int): Number of copies of the style manager to create.
            use_cuda_streams (bool): Whether to use CUDA streams for parallelism.
        """
        self.style_managers = {"art": [], "photo": []}
        num_copies_list = [num_copies, num_copies] if isinstance(num_copies, int) else list(num_copies)
        self.total_copies = sum(num_copies_list)
        self.style_transfer_semaphore = {key: Semaphore(num_copies_list[idx]) for idx, key in enumerate(self.style_managers.keys())}
        self.style_manager_lock = Lock()
        self.use_cuda_streams = use_cuda_streams
        # preallocate style managers
        self._initialize_style_managers(transfer_type, ckpt_dir, max_size, num_copies_list)
        # optionally create CUDA streams for each manager if true
        self.cuda_streams = [torch.cuda.Stream() for _ in range(self.total_copies)] if self.use_cuda_streams else None
        if self.use_cuda_streams:
            torch.cuda.synchronize()
        # track available managers using a dictionary with two separate queues
        self.available_managers = {key: Queue() for key in self.style_managers.keys()}
        for mode_idx, mode in enumerate(self.available_managers.keys()):  # for each mode
            for i in range(num_copies_list[mode_idx]):
                self.available_managers[mode].put(i)  # Each manager's index will be added to the queue


    def _initialize_style_managers(self, transfer_type, ckpt_dir, max_size, num_copies_each: List[int]):
        raise DeprecationWarning("This method is deprecated and will be removed in a future version")
        if not isinstance(transfer_type, (list, tuple)):
            transfer_type = [transfer_type]
        # TODO: need a new way of loading and filtering configuration arguments before just calling .infer.stage_inference_pipeline
        for mode_idx, mode in enumerate(transfer_type):
            for _ in range(num_copies_each[mode_idx]):
                ckpt_path = os.path.join(ckpt_dir, f"{mode}_image.pt")
                # FIXME: definitely don't want to save results here - use temporary hard-coded absolute path later
                st_output_dir = os.path.join(ckpt_dir, "style_transfer")
                manager = BaseImageStylizer(mode=mode, ckpt=ckpt_path, max_size=max_size,
                                                        save_results=False, output_dir=st_output_dir)
                self.style_managers[mode].append(manager)


    def get_available_manager(self, mode="art"):
        """ Acquires an available style manager and its index from the pool.
            Args:
                mode (str): The mode for style transfer, either 'art' or 'photo'.
            Returns:
                Tuple: (Style manager, index)
        """
        # Wait until a manager is available
        self.style_transfer_semaphore[mode].acquire()
        # Safely get an available manager's index
        manager_index = self.available_managers[mode].get()
        # Lock to ensure thread-safety for assigning a manager
        with self.style_manager_lock:
            return self.style_managers[mode][manager_index], manager_index

    def release_manager(self, manager_index, mode="art"):
        """ Releases the style manager back to the pool after use.
            Args:
                manager_index (int): The index of the style manager to release.
        """
        # Mark the manager as available again
        with self.style_manager_lock:
            self.available_managers[mode].put(manager_index)
        # Release the semaphore to signal availability
        self.style_transfer_semaphore[mode].release()


    def transform(self, sample, style_paths, mode="art", **kwargs):
        """ Performs the style transfer using an available manager, either in parallel or sequentially.
            Args:
                sample (torch.Tensor): Input sample to apply style transfer to.
                style_paths (list): List of paths to style images.
                mode (str): The mode for style transfer ('art' or 'photo').
                **kwargs: Additional arguments passed to the style manager's transform method.
            Returns:
                torch.Tensor: Stylized image.
        """
        # Get an available style manager
        manager, manager_index = self.get_available_manager(mode)
        try:
            # Use CUDA streams if enabled
            if self.use_cuda_streams:
                #self.cuda_streams[manager_index].wait_stream(torch.cuda.current_stream())
                idx = manager_index if mode == "art" else manager_index + len(self.style_managers["art"])
                with torch.cuda.stream(self.cuda_streams[idx]):
                    # Perform style transfer asynchronously
                    return manager.transform(sample, style_paths, **kwargs)
                #torch.cuda.current_stream().wait_stream(self.cuda_streams[manager_index])
                torch.cuda.synchronize()
            else:
                # Perform style transfer without streams
                return manager.transform(sample, style_paths, **kwargs)
        finally:
            # Ensure the manager is released back to the pool after use
            self.release_manager(manager_index, mode)


    def __repr__(self):
        return f"StyleTransferDispatcher(managers={self.total_copies}, use_cuda_streams={self.use_cuda_streams})"


# ? NOTE: to use this with the non-policy model pipeline, I should probably make a separate method to call in augment_batch to not use CUDA streams on top of those used above


"""
Key Features of StyleTransferDispatcher
Manager Pooling:

Multiple BaseImageStylizer instances are preallocated and stored in a pool.
The Semaphore ensures that only a limited number of managers are in use at any given time.
Asynchronous Execution with CUDA Streams:

If use_cuda_streams is enabled, each style transfer task is executed in its own CUDA stream, allowing asynchronous parallelism.
Locking and Availability Management:

The Semaphore and Lock control access to the style managers, ensuring that only one task can use a manager at a time.
The Queue tracks available managers and ensures that tasks are assigned to an available manager in a thread-safe manner.
Integration with Existing Pipeline:

The transform method handles style transfer requests, so it can easily be passed to AugmentationFunctionalWrapper.handle in the augmentation pipeline.


Benefits of this Design
Concurrency Without Race Conditions: By cloning style transfer managers and managing them through a semaphore, you ensure multiple parallel style transfers without race conditions.
Asynchronous CUDA Streams: If you’re using CUDA, the asynchronous stream logic ensures that the GPU is efficiently utilized for parallel style transfer operations.
Minimal Code Changes: You can keep the existing augmentation interface in place by passing the StyleTransferDispatcher.transform method to AugmentationFunctionalWrapper.
"""