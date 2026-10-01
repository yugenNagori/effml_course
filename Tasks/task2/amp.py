import torch
import torch.nn as nn
import torch.nn.functional as F

#         Maintain a primary copy of weights in FP32.
# For each iteration:
# Make an FP16 copy of the weights.
# Forward propagation (FP16 weights and activations).
# Multiply the resulting loss with the scaling factor S.
# Backward propagation (FP16 weights, activations, and their gradients).
# Multiply the weight gradient with 1/S.
# Complete the weight update (including gradient clipping, etc.).
class Autocast:

    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.target_dtype = dtype
        self._original_funcs = {}
        self._depth = 0

        # Define the functions we want to intercept and cast.
        # Format: (module, function_name_string)
        self.DOWNCAST_OPS = [
            (torch, 'matmul'),
            (torch, 'bmm'),
            (F, 'linear'),
        ]
        self.UPCAST_OPS = [
            (F, 'layer_norm'),
            (F, 'cross_entropy'),
        ]


    def _create_wrapper(self, original_func, target_dtype):
        """
        Creates a wrapper that casts inputs to the target dtype
        and calls the function with casted arguments
        """
        def wrapper(*args, **kwargs):
            """
            Wrapper that casts all inputs to the target dtype 
            and calls the function with casted arguments
            """

            # Cast all tensor arguments in args and kwargs
            new_args = []
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    if torch.is_floating_point(args[i]):
                        new_args.append(args[i].to(target_dtype))
                    else:
                        new_args.append(args[i])
                else:
                    new_args.append(args[i])
    
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    if torch.is_floating_point(kwargs[key]):
                        kwargs[key] = kwargs[key].to(target_dtype)

            # Call the original function with potentially casted inputs
            return original_func(*new_args, **kwargs)
            

        return wrapper

    def __enter__(self):
        """
        Wraps all the functions from DOWNCAST_OPS and UPCAST_OPS,
        Stores original functions
        And sets wrapped functions instead of original ones in the module
        """
        if not self.enabled:
            return self

        self._depth += 1
        if self._depth > 1:
            return self

        # Store original functions and apply patches
        for module, func_name in self.DOWNCAST_OPS + self.UPCAST_OPS:
            # Store original function
            together = (module, func_name)
            curr_func = getattr(*together)
            self._original_funcs[together] = curr_func

            # Create wrapped version of the function
            # Note that you need different target_dtype for DOWNCAST_OPS and UPCAST_OPS
            needed_dtype = torch.float32 if together in self.UPCAST_OPS else self.target_dtype
            new_func = self._create_wrapper(curr_func, needed_dtype)

            # Set wrapped function as attribute of the module with the same name as original function
            setattr(*together, new_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores original function
        """
        if not self.enabled:
            return

        self._depth -= 1
        if self._depth > 0:
            return False

        # Restore original functions
        for key, func in self._original_funcs.items():
            setattr(*key, func)
        
        # Clear the stored functions for the next use
        self._original_funcs.clear()
        return False


class StaticGradScaler:
    def __init__(self, scale):
        """
        scale: loss scaling coef
        """
        self.scale_coef = scale
    
    def scale(self, loss):
        """Scales the loss"""
        scaled_loss = loss * self.scale_coef
        return scaled_loss
    
    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        flag_nan = False
        for gr_param in optimizer.param_groups:
            if flag_nan: break

            for param in gr_param['params']:
                if param.grad is None:
                    continue
                else:
                    if not torch.isfinite(param.grad).all():
                        flag_nan = True
                        break

        if not flag_nan:
            for gr_param in optimizer.param_groups:
                for param in gr_param['params']:
                    if param.grad is None:
                        continue
                    else:
                        param.grad /= self.scale_coef
            optimizer.step()
    
        
        
    
    def update(self):
        """Updates scaling coef"""
        pass


class DynamicGradScaler:
    def __init__(self, scale, factor, patience, min_scale, max_scale):
        """
        scale: initial value of loss scaling coef
        factor: multiplier that used for scaling coef increase/decrease
        patience: how many iters there should be no nan/inf to increase scale
        min_scale: minimal allowed scaling coef value
        max_scale: maximal allowed scaling coef value
        """
        self.scale_coef = scale
        self.factor = factor
        self.patience = patience
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.steps_without_nan = 0
        self.flag_nan = False

    def scale(self, loss):
        """Scales the loss"""
        scaled_loss = loss * self.scale_coef
        return scaled_loss

    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # If there is any nan/inf in the gradient decrease scaling coef using factor
        # Note that scaling coef should be greater than min_scale
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        self.flag_nan = False
        for gr_param in optimizer.param_groups:
            if self.flag_nan: break

            for param in gr_param['params']:
                if param.grad is None:
                    continue
                else:
                    if not torch.isfinite(param.grad).all():
                        self.flag_nan = True
                        self.steps_without_nan = 0
                        self.scale_coef /= self.factor
                        self.scale_coef = max(self.scale_coef, self.min_scale)
                        break

        if not self.flag_nan:
            self.steps_without_nan += 1

            for gr_param in optimizer.param_groups:
                for param in gr_param['params']:
                    if param.grad is None:
                        continue
                    else:
                        param.grad /= self.scale_coef
            optimizer.step()
    

    def update(self):
        """Updates scaling coef"""
        # If there was no any nan/inf in the gradient patience steps, increase scaling coef using factor
        # Note that scaling coef should be smaller than max_scale
        if not self.flag_nan:
            if self.steps_without_nan >= self.patience:
                self.scale_coef *= self.factor
                self.steps_without_nan = 0

        self.scale_coef = min(self.scale_coef, self.max_scale)
