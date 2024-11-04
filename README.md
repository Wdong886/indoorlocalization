# indoorlocalization
we propose an indoor visual localization method based on image-point cloud cluster matching. First, the acquired 3D point cloud model of the indoor scene is segmented into multiple clusters, and an image–point cloud cluster database is constructed to enable rapid, coarse position estimation of the query image based on its visual features. Second, cross-modal matching between the image and point cloud clusters is performed using image diffusion features extracted by the deep generative model Stable Diffusion. Finally, false matches in the cross-modal point pairs are filtered out, and the PnP algorithm is applied to estimate personnel location accurately.Experimental results demonstrate that the proposed method achieves an average query time of 6.07 seconds for rough position estimation. The cross-modal matching accuracy and recall are 69.24% and 56.38%, respectively, outperforming traditional methods such as SIFT+FPFH and I2PC. Additionally, the root-mean-square error (RMSE) of localization is 1.136 meters. 

#Related Projects
We sincerely thank the excellent projects:

[IDC-DC for depth completion.](https://github.com/kujason/ip_basic)

https://github.com/AUTOMATIC1111/stable-diffusion-webui
