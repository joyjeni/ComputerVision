![segmentation](https://i.imgur.com/LCxrHQ7.png)


1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention 
***(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)***
The encoded image is taken from the DETR Encoder Output, by default d=256.CNN backbone outputs low resolution feature map of H/32, W/32. This output is matched with the hidden dimensions of DETR (256 default) and then sent to encoder. The output of the encoder has image features of the same dimension. This output is fed as input to the Multi head attention for panoptic segmentation.


2. We also send dxN Box embeddings to the Multi-Head Attention
We do something here to generate NxMxH/32xW/32 maps.
***(WHAT DO WE DO HERE?)***

We use multi-head attention layer to generate attention scores over the encoded image (dxH/32xW/32) for each Object Embedding (N). 
This results in NxMxH/32xW/32 attention maps.

3. Then we concatenate these maps with Res5 Block 

***(WHERE IS THIS COMING FROM?)***


This comes from the CNN Backbone of DETR, when the Image is passed by CNN Backbone, activations from the intermediate layers (Res 5, Res4, Res3 ad Res 2) are set aside so that it can be used while doing panoptic segmentation.

4. Then we perform the above steps (EXPLAIN THESE STEPS) And then we are finally left with the panoptic segmentation

    
    1. The model predicts a box and a binary mask for each object queries. Filter the predictions for which the confidence is < 85%
    2. These BBox along with Encoded Image (from the Encoder Output) is sent to the Multi Head Attention Layer.
    3. This Layer generates the Attention Maps for the passed Bboxes using Encoded Image.
    4. These Attention Maps are concatenated with Res 5 (output of CNN Backbone Intermediate Layer) and various operations are performed to Upsample
       and Clean the Masks using    intermediate layers output of the CNN backbone.
    5. Finally masks are merged together using a pixel-wise argmax. Output image is the Panoptic Segmentation of the Original Image with dimension 1/4 of the Original Image.

5. Details on the solution  https://github.com/joyjeni/ComputerVision/blob/main/capestone/panopticsegmentation.md




