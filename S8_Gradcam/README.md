
### Developers

1. Vivek K  
2. Jenisha Thankaraj

### Model Diagnostics using Gradient-weighted class activation mapping(Gradcam)

1. Choose a convolutional layer for visualization
2. Get output for the layer specified after finding error
3. Find gradients for the layer specified
4. For both remove batch dimension not its only 3d(h,w,channel)
5. Convert gradients to weights by taking mean along 0,1(height, width) axis
6. reshape channel to image shape  to overlay in input image
7. compute np.dot(output,weight)
8. Normalize 
9. Create heatmap
10. Overlay healhap on the image



