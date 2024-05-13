import numpy as np
class SEGFACADEDataset():
    num_labels = 12
    
    unique_colors = [
        [  0],
        [  110],
        [  240],
        [  200],
        [  250],
        [  230],
        [  80],
        [  60],
        [  140],
        [  160],
        [  180],
        [  30],
    ]
    
    id2label = {
        0: "szum",
        1: "drzwi",
        2: "okno", 
        3: "portal", 
        4: "obramowanie_okienne", 
        5: "naczolki_okienne", 
        6: "gzyms/element_poziomy", 
        7: "pilaster/element podzialu pionowy", 
        8: "balkon",  
        9: "dach",
        10: "detal", 
        11: "sciana"
    }

    def parse_segmentation_image(self, mask_image):
        """
        Map R pixel values in the RGB segmentation image to class ids.

        Args:
            segmentation_image: a WxHx3 numpy array representing the segmentation image

        Returns:
            A WxH numpy array where each element is the class id for the corresponding
            pixel in the segmentation image.
        """

        segmentation_image = np.array(mask_image).copy()
        # Create a mapping from RGB values to class ids
        color_to_class = {tuple(color): i for i, color in enumerate(self.unique_colors)}

        # Initialize the output array with the correct number of classes
        output = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), dtype=np.uint8)

        # Map the RGB values to class ids
        for i in range(segmentation_image.shape[0]):
            for j in range(segmentation_image.shape[1]):
                color = segmentation_image[i, j, 0]
                try:
                    output[i, j] = color_to_class[(color,)]
                except KeyError:
                    output[i, j] = color_to_class[(0,)]


        return output