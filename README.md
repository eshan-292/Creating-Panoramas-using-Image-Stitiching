**Approach:**

Image Registration -
I implemented a SIFT based feature detector to detect the keypoint and their 128
dimensional decriptors. It involved approximating the Laplacian of Gaussian (LoG) by
using the Difference of Gaussians (DoG) to address the problem of scale invariance
Then I found the local extrema in a 3x3x3 neighborhood to detect key-points. Followed
by assigning each key-point an orientation according to the surrounding neighborhood.
Based on this I created an orientation histogram by quantizing the gradient orientations
within a region around the key-point. Next, to generate the descriptor, I created a 16x16
neighborhood centered at the key-point, which is further divided into smaller cells of size
4x4. Then I computed orientation histograms for each cell and concatenated them into a
single 128-dimensional feature descriptor, providing a compact representation of the
key-point's local appearance and orientation.
I tried the ORB based feature detector too, and although it was faster in computation but
it provided low accuracy in detecting useful features thereby leading to poor reults.
I also experimented with the intensity based methods like thex Normalized
Cross-Correlation but they were computationally expensive to compute and were not
leading to any significant change in the results.


Panorama Creation -
For stitching together two images, I matched the SIFT keypoints of both the images
using a brute force based approach which finds the best matching keypoint in image 2
for each keypoint in image 1. I also tried the k nearest neighbours algorithm for
matching the keypoints, but that was leading to quite less number of matchings so I
stuck with the Brute force based one. Then I filtered the matchings based on the angles,
that is the matchings which were more than 20 degrees to the x axis were removed
(because we do not want too much distortion). Also, I restricted the matchings to the
one half where the images were supposed to overlap. This led to much better matching
detection.

Then I used the matchings to compute the homography between the two images. For
this I implemented a RANSAC based method by iteratively selecting a random subset of
correspondences, computing a homography (using the SVD technique), and keeping
the homography that yields the maximum number of inliers.

Then I warped the images using the found homography matrix by applying a
perspective transformation and then using bilinear interpolation to determine the pixel
value in the output image. Then I simply overlaid the warped images onto the common
coordinate system to generate the output.

I also implemented alpha blending and although it was simple and fast but it led to very
visible borders and seams between the images and was worse than just superimposing.
Then I also implemented the laplacian blending which was time consuming and
computationally expensive, but it led to a better smoothened comniation of the images.
However it had one serious issue, that it led to gaussian smoothing of the image and
then when I had to combine the panorama with the next set of images it led to very poor
detection of the keypoints and matching. Hence I had to drop this idea and just work
with the superimposition one.


Videos:
For generating the panoramas for videos, I simply extracted 6 equally distributed frames
from the video, added the last frame to ensure that the complete scene is captured and
fed those images to the image stitcher to generate a panorama
