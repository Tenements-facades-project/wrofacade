from src.facade_generator.facade_generator_abc import FacadeGenerator
import torch 
from torch.distributions.normal import Normal
from src.utils.imgtrans_utils.image_processing import tensor_to_img

class GANGeneratorTorch(FacadeGenerator):
    def __init__(self, model:torch.nn.Module, hp: dict, input_vector_transform = None, forward_decorator = None):
        """Initialize the TransSegmentationHF class.

        Parameters:
            model - model with randomly initiated parametres with the
                    target architecture
            hp - dictionary containing parametres with next entries:
                'checkpoint_path':str  -- string containing path to the checkpoint file,
                            to load model's parametres from

                'input_dim':int  -- dimention of the vector, expected as an input to the model

                'distribution':string -- name of the distribution to sample vectors from

                'device':torch.device|str -- torch.device object. If not provided will use CUDA if avaliable,
                        if not CPU will be used

            input_vector_transform - lambda function that converts sampled input vector ([B, H] tensor)
                                     into format model expects to accept

            forward_decorator - lambda function to apply to the model's forward function result that
                                converts the result to the expected format ([1, Channels, Height, Width] tensor) 
        """
        self.model = model.eval()

        self.checkpoint_path = hp['checkpoint_path']

        self.device = hp['device']

        self.input_dimention = hp['input_dim']

        self.distribution = hp['distribution']

        self.input_vector_transform = input_vector_transform

        self.forward_decorator = forward_decorator

        self.model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))


        if self.distribution == 'normal':
            self.dist = Normal(
                torch.tensor([0.0] * self.input_dimention),
                torch.tensor([1.0] * self.input_dimention),
                validate_args=None,
            )
        else:
            raise Exception(f'Unknown distribution: {self.distribution}')

    def generate_facade(self):
        """
        TODO: add sampling
        """

        input_tensor = self.dist.sample().unsqueeze(0)
        
        if self.input_vector_transform is not None:
            input_tensor = self.input_vector_transform(input_tensor)

        result = self.model(input_tensor)
        
        if self.forward_decorator is not None:
            result = self.forward_decorator(result)
        
        return tensor_to_img(result)
