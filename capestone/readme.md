![segmentation](https://i.imgur.com/LCxrHQ7.png)


1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention 
***(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)***

The Encoded Image is taken from the DETR Encoder Output, by default d=256.

CNN Backbone of the DETR outputs a Low Resolution Feature Map of h/32 and w/32. This output is matched with the Hidden Dimensions of DETR (256 default) and then sent to Encoder. The Output of the Encoder is Image Features of the same Dimension. This Output is used as Encoded Image (one of the Inputs) to the Multi Head Attention for Panoptic Segmentation.

2. We also send dxN Box embeddings to the Multi-Head Attention
We do something here to generate NxMxH/32xW/32 maps.
***(WHAT DO WE DO HERE?)***

We use Multi-head attention layer to generate Attention Scores over the Encoded Image (dxH/32xW/32) for each Object Embedding (N). 
This results in NxMxH/32xW/32 attention maps.

3. Then we concatenate these maps with Res5 Block 
***(WHERE IS THIS COMING FROM?)***

This comes from the CNN Backbone of DETR, when the Image is passed by CNN Backbone, Activations from the Intermediate layers (Res 5, Res4, Res3 ad Res 2) are set aside so that it can be used while doing Panoptic Segmentation.

4. Then we perform the above steps (EXPLAIN THESE STEPS) And then we are finally left with the panoptic segmentation

    1. DETR Predicts the BBox for thing (your class) and Stuffs in the image.
    2. These BBox along with Encoded Image (from the Encoder Output) is sent to the Multi Head Attention Layer.
    3. This Layer generates the Attention Maps for the passed Bboxes using Encoded Image.
    4. These Attention Maps are concatenated with Res 5 (output of CNN Backbone Intermediate Layer) and various operations are performed to Upsample
       and Clean the Masks using    intermediate layers output of the CNN backbone.
    5. Finally we merge the masks by classigying each pixel to the mask with highest Probability.
    6. Resultant image is the Panoptic Segmentation of the Original Image with dimension 1/4 of the Original Image.

5. Details on the solution  https://github.com/joyjeni/ComputerVision/blob/main/capestone/panopticsegmentation.md




