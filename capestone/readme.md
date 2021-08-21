![segmentation](https://i.imgur.com/LCxrHQ7.png)


1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention 
***(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)***






2. We also send dxN Box embeddings to the Multi-Head Attention
We do something here to generate NxMxH/32xW/32 maps.
***(WHAT DO WE DO HERE?)***


3. Then we concatenate these maps with Res5 Block 
***(WHERE IS THIS COMING FROM?)***

4. Then we perform the above steps (EXPLAIN THESE STEPS) And then we are finally left with the panoptic segmentation

5. Details on the solution  


