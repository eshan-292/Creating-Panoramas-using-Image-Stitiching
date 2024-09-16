
import cv2
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from helper import computeKeypointsAndDescriptors, brute_force_matcher, ransac, warpPerspective, detect_and_match_features_lin, get_new_frame_size_and_homography_lin





def preprocess_image(image):
    # Apply contrast adjustment, noise reduction, and color correction
    # You can use OpenCV functions like cv2.equalizeHist(), cv2.fastNlMeansDenoising(), cv2.cvtColor() for this purpose
    # Example:
    # processed_image = cv2.equalizeHist(image)

    # apply gaussian noise reduction
    # image = cv2.GaussianBlur(image, (5, 5), 1)

    return image



def detect_and_match_features(image1, image2, direction):
    

    # # # Detect keypoints and compute descriptors using SIFT
    # sift = cv2.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # # convert keypoints to tuple
    # keypoints1 = [(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints1]
    # keypoints2 = [(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints2]



    # use custom SIFT
    # convert the images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = computeKeypointsAndDescriptors(image1_gray)
    keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2_gray)

    # # load the keypoints and descriptors
    # keypoints1 = pickle.load(open("data/keypoints1.pkl", "rb"))
    # descriptors1 = pickle.load(open("data/descriptors1.pkl", "rb"))
    # keypoints2 = pickle.load(open("data/keypoints2.pkl", "rb"))
    # descriptors2 = pickle.load(open("data/descriptors2.pkl", "rb"))
    # # convert the keypoints back to original format
    # keypoints1 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints1]
    # keypoints2 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints2]

   
    # bf = cv2.BFMatcher()
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key = lambda x:x.distance)

    # # print the length of the matches
    # print("Number of matches:", len(matches))

    # new_matches = []
    # # remove the matches such that the angle wrt x axis is greater than 30 degrees
    # for match in matches:
    #     x1, y1 = keypoints1[match.queryIdx][:2]
    #     x2, y2 = keypoints2[match.trainIdx][:2]

    #     angle = np.arctan((y2 - y1) / (x2 - x1))
    #     angle = np.degrees(angle)
    #     if abs(angle) < 20:
    #         new_matches.append(match)
        
    # sorted(new_matches, key = lambda x:x.distance)



    matches = brute_force_matcher(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)
    print("Number of matches:", len(matches))
    new_matches = []
    # remove the matches such that the angle wrt x axis is greater than 30 degrees
    for match in matches:
        x1, y1 = keypoints1[match.queryIdx][:2]
        x2, y2 = keypoints2[match.trainIdx][:2]
        angle = np.arctan((y2 - y1) / (x2 - x1))
        angle = np.degrees(angle)
        if abs(angle) < 20:
            new_matches.append(match)

    # sort the matches
    new_matches = sorted(new_matches, key=lambda match: match.distance)

    
    
    
    good_matches = new_matches[:min(1000, len(new_matches))]
    # src_pts = np.array([keypoints1[match.queryIdx][:2] for match in good_matches])
    # dst_pts = np.array([keypoints2[match.trainIdx][:2] for match in good_matches])



    # if the direction is left, then remove all the matches that are not on the left side of the image
    if direction == 'left':
        center = image1.shape[1] // 2
        # remove all the matches that are on the right side of the image
        good_matches = [m for m in good_matches if keypoints1[m.queryIdx][0] <= center]
    elif direction == 'right':
        center = image1.shape[1] // 2
        # remove all the matches that are on the left side of the image
        good_matches = [m for m in good_matches if keypoints1[m.queryIdx][0] >= center]
        
    
# print number of final matches
    print("Number of final matches:", len(good_matches))

    correspondences = []
    for match in good_matches:
        x1, y1 = keypoints1[match.queryIdx][:2]
        x2, y2 = keypoints2[match.trainIdx][:2]
        
        # correspondences.append([x1, y1, x2, y2])
        correspondences.append([x2, y2, x1, y1])

    correspondences = np.matrix(correspondences)
    


        
    
    
    
    # convert the keypoints to the cv2 keypoints for visualization
    keypoints1 = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints1]
    keypoints2 = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints2]
    # convert the good matches to the cv2 matches
    good_matches = [cv2.DMatch(m.queryIdx, m.trainIdx, m.distance) for m in good_matches]   
    
    # Show the matched keypoints
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matched Keypoints", matched_image)
    # cv2.waitKey(0)
    
    
    # run ransac
    H_custom, inliers = ransac(correspondences, 5)
    # convert to numpy.ndarray
    H_custom = np.array(H_custom)
    
    # # print H_custom
    # print("Homography CUstom:", H_custom)

    # # Extract matched keypoints
    # src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    
    # H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    # # print H
    # print("Homography OpenCV:", H)

    # return H
    # return src_pts, dst_pts




    
def Lucas_Kanade_Optical_Flow(image1, image2):
    # Implement Lucas-Kanade Optical Flow algorithm to track the movement of keypoints between two images
    # You can use OpenCV function cv2.calcOpticalFlowPyrLK() for this purpose
    # Example:
    p0 = cv2.goodFeaturesToTrack(image1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p1, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p0, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return p0, p1


    

def calculate_frame_size_and_update_homography(image2, image1, homography):
    # Get the size of the second image
    height1, width1 = image2.shape[:2]

    # Define the initial coordinates of the corners of the second image
    initial_coordinates = np.array([[0, width1-1, width1-1, 0], [0, 0, height1-1, height1-1], [1, 1, 1, 1]])

    # Apply the homography to the initial coordinates
    final_coordinates = np.dot(homography, initial_coordinates)

    # Normalize the coordinates
    x, y, z = final_coordinates
    x = np.divide(x, z)
    y = np.divide(y, z)

    # Calculate the dimensions of the new frame and the correction factor
    x_min = int(round(min(x)))
    x_max = int(round(max(x)))
    y_min = int(round(min(y)))
    y_max = int(round(max(y)))

    new_width = x_max
    new_height = y_max
    correction = [0,0]
    if x_min < 0:
        new_width -= x_min
        correction[0] = -x_min
    if y_min < 0:
        new_height -= y_min
        correction[1] = -y_min

    # Correct the new width and height if the second image overlaps on the left side of the first image
    if new_width < image1.shape[1] + correction[0]:
        new_width = image1.shape[1] + correction[0]
    if new_height < image1.shape[0] + correction[1]:
        new_height = image1.shape[0] + correction[1]

    # Adjust the coordinates of the corners of the second image
    x = np.add(x, correction[0]) 
    y = np.add(y, correction[1])
    old_points = np.float32([[0, 0], [width1-1, 0], [width1-1, height1-1], [0, height1-1]])

    final_points = np.float32(np.array([x, y]).T)

    # Update the homography matrix
    correspondences = []
    for i in range(4):
        correspondences.append([old_points[i][0], old_points[i][1], final_points[i][0], final_points[i][1]])
    correspondences = np.matrix(correspondences)
    H, _ = ransac(correspondences, 5)
    homography = np.array(H)

    return [int(new_height), int(new_width)], correction, homography
        
        
        

def convert_coordinates(x, y, center, focal_length):
    x_transformed = (focal_length * np.tan((x - center[0]) / focal_length)) + center[0]
    y_transformed = ((y - center[1]) / np.cos((x - center[0]) / focal_length)) + center[1]
    return x_transformed, y_transformed

def project_onto_cylinder(image):
    height, width = image.shape[:2]
    center = [width // 2, height // 2]
    global focal_length
    # focal_length = 1100  # Focal length, adjust based on specific requirements
    
    # Initialize a blank transformed image
    transformed_image = np.zeros(image.shape, dtype=np.uint8)
    
    # Store all coordinates of the transformed image in 2 arrays (x and y coordinates)
    all_coordinates = np.array([np.array([i, j]) for i in range(width) for j in range(height)])
    transformed_x = all_coordinates[:, 0]
    transformed_y = all_coordinates[:, 1]
    
    # Find corresponding coordinates of the transformed image in the initial image
    initial_x, initial_y = convert_coordinates(transformed_x, transformed_y, center, focal_length)

    # Round off the coordinate values to get exact pixel values (top-left corner)
    initial_tl_x = initial_x.astype(int)
    initial_tl_y = initial_y.astype(int)

    # Find transformed image points whose corresponding initial image points lie inside the initial image
    valid_indices = (initial_tl_x >= 0) * (initial_tl_x <= (width-2)) * \
                    (initial_tl_y >= 0) * (initial_tl_y <= (height-2))

    # Remove all the outside points
    transformed_x = transformed_x[valid_indices]
    transformed_y = transformed_y[valid_indices]
    initial_x = initial_x[valid_indices]
    initial_y = initial_y[valid_indices]
    initial_tl_x = initial_tl_x[valid_indices]
    initial_tl_y = initial_tl_y[valid_indices]

    # Bilinear interpolation
    dx = initial_x - initial_tl_x
    dy = initial_y - initial_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = dx * (1.0 - dy)
    weight_bl = (1.0 - dx) * dy
    weight_br = dx * dy
    
    transformed_image[transformed_y, transformed_x, :] = (weight_tl[:, None] * image[initial_tl_y,     initial_tl_x,     :]) + \
                                                         (weight_tr[:, None] * image[initial_tl_y,     initial_tl_x + 1, :]) + \
                                                         (weight_bl[:, None] * image[initial_tl_y + 1, initial_tl_x,     :]) + \
                                                         (weight_br[:, None] * image[initial_tl_y + 1, initial_tl_x + 1, :])

    # Remove black region from right and left in the transformed image
    min_x = min(transformed_x)

    # Crop out the black region from both sides (using symmetry)
    transformed_image = transformed_image[:, min_x : -min_x, :]

    return transformed_image, transformed_x-min_x, transformed_y


def combine_images(image1, image2, direction, projection):
    
    # ###### With Cylinderical Projection
    if projection == 'cyl':

        # Applying Cylindrical projection on imag2
        image2_cyl, mask_x, mask_y = project_onto_cylinder(image2)

        # # show image2_cyl
        # cv2.imshow("Cylindrical Image 2", image2_cyl)
        # cv2.waitKey(0)
        

        # # Getting  Mask for image2
        image_2_mask = np.zeros(image2.shape, dtype=np.uint8)
        image_2_mask[mask_y, mask_x, :] = 255

        # Finding the matches
        # src_pts, dst_pts = detect_and_match_features(image1, image2_cyl, direction)

        # homography = estimate_homography(dst_pts, src_pts)
        homography   = detect_and_match_features(image1, image2_cyl, direction)

        

        # get new frame size and homography
        new_frame_size, correction, homography = calculate_frame_size_and_update_homography(image2_cyl, image1, homography)


        # calc the condition number of the homography first two columns
        print("Condition number of the homography:", np.linalg.cond(homography[:, :2]))
        condition_number = np.linalg.cond(homography[:, :2])
        if condition_number > 2.0:
            print("The condition number of the homography is greater than 2.0")
            flag =1

        # placing the images on the frame
        # image2_warped = cv2.warpPerspective(image2_cyl, homography, (new_frame_size[1], new_frame_size[0]))
        image2_warped = warpPerspective(image2_cyl, homography, new_frame_size)

        # show the warped image
        # cv2.imshow("Warped Image 2", image2_warped)
        # cv2.waitKey(0)

        
        # image2_transformed_mask = cv2.warpPerspective(image_2_mask, homography, (new_frame_size[1], new_frame_size[0]))
        image2_transformed_mask = warpPerspective(image_2_mask, homography, new_frame_size)
        
        image1_transformed = np.zeros((new_frame_size[0], new_frame_size[1], 3), dtype=np.uint8)
        image1_transformed[correction[1]:image1.shape[0]+correction[1], correction[0]:image1.shape[1]+correction[0]] = image1
        
        # # show image1 transformed
        # cv2.imshow("Image 1 Transformed", image1_transformed)
        # cv2.waitKey(0)


        # panorama = cv2.bitwise_or(image2_warped, cv2.bitwise_and(image1_transformed, cv2.bitwise_not(image2_transformed_mask)))
        panorama = np.bitwise_or(image2_warped, np.bitwise_and(image1_transformed, np.bitwise_not(image2_transformed_mask)))

        # panorama = straighten_panorama(panorama)


        #  apply linear blending in the region of overlap
        # use the mask to blend the images using alpha blending
        # stitched_image = cv2.addWeighted(image1_transformed, 0.5, image2_warped, 0.5, 0)


        # apply laplacian blending
        # stitched_image = laplacian_blending(image1_transformed, image2_warped)

        return panorama

    
    ##### Without Cylindrical Projection

    else:

        flag = 0
        # # Getting  Mask for image2
        image_2_mask = np.zeros(image2.shape, dtype=np.uint8)
        image2_cyl, mask_x, mask_y = project_onto_cylinder(image2)
        image_2_mask[mask_y, mask_x, :] = 255
        # homography = estimate_homography(dst_pts, src_pts)
        homography   = detect_and_match_features_lin(image1, image2_cyl, direction)

        # calc the condition number of the homography first two columns
        print("Condition number of the homography:", np.linalg.cond(homography[:, :2]))
        condition_number = np.linalg.cond(homography[:, :2])
        if condition_number > 2.0:
            print("The condition number of the homography is greater than 2.0")
            flag =1


        # get new frame size and homography
        new_frame_size, correction, homography = get_new_frame_size_and_homography_lin(image2_cyl, image1, homography)

        # placing the images on the frame
        image2_warped = cv2.warpPerspective(image2_cyl, homography, (new_frame_size[1], new_frame_size[0]))
        # image2_warped = warpPerspective(image2_cyl, homography, new_frame_size)

        # show the warped image
        # cv2.imshow("Warped Image 2", image2_warped)
        # cv2.waitKey(0)

        
        image2_transformed_mask = cv2.warpPerspective(image_2_mask, homography, (new_frame_size[1], new_frame_size[0]))
        # image2_transformed_mask = warpPerspective(image_2_mask, homography, new_frame_size)
        
        image1_transformed = np.zeros((new_frame_size[0], new_frame_size[1], 3), dtype=np.uint8)
        image1_transformed[correction[1]:image1.shape[0]+correction[1], correction[0]:image1.shape[1]+correction[0]] = image1
        
        # # show image1 transformed
        # cv2.imshow("Image 1 Transformed", image1_transformed)
        # cv2.waitKey(0)


        # panorama = cv2.bitwise_or(image2_warped, cv2.bitwise_and(image1_transformed, cv2.bitwise_not(image2_transformed_mask)))
        panorama = np.bitwise_or(image2_warped, np.bitwise_and(image1_transformed, np.bitwise_not(image2_transformed_mask)))


       
        #  apply linear blending in the region of overlap
        # use the mask to blend the images using alpha blending
        # stitched_image = cv2.addWeighted(image1_transformed, 0.5, image2_warped, 0.5, 0)


        # apply laplacian blending
        # stitched_image = laplacian_blending(image1_transformed, image2_warped)

        return (panorama, flag)






    
    


def stitch_images(image_list, mode='image', projection='cyl'):
    # Preprocess images
    processed_images = [preprocess_image(image) for image in image_list]
    # resize all the images to the same size
    # processed_images = [cv2.resize(image, (1200, 1400)) for image in processed_images]

    new_images = []
    cnt =1
    g_flag = 0
    # iterate starting from the middle and expanding one at a time in both directions
    # to get the final stitched image
    
    if projection == 'cyl':
        
        panorama, _, _ = project_onto_cylinder(processed_images[len(processed_images)//2])
    else:
        panorama = processed_images[len(processed_images)//2]
    projection = 'cyl' if projection == 'lin' else 'lin' if projection == 'cyl' else projection
    for i in range(1, len(processed_images)//2 + 1):
          # check if i is in the range of the images
        if len(processed_images)//2 -i >= 0:
            if mode == 'image':
                (stitched_image, flag) = combine_images(panorama, processed_images[len(processed_images)//2 - i], 'left', projection)
                if flag==1:
                    g_flag = 1
            else:
                (stitched_image, flag) = combine_images(panorama, processed_images[len(processed_images)//2 - i], 'vid', projection)
                if flag==1:
                    g_flag = 1
            
            
            panorama = stitched_image.copy()
            # show the panorama
            # cv2.imshow("Panorama", panorama)
            # cv2.waitKey(0)



        if len(processed_images)//2 + i < len(processed_images):
            if mode == 'image':
                (stitched_image, flag) = combine_images(panorama, processed_images[len(processed_images)//2 + i], 'right', projection)
                if flag==1:
                    g_flag = 1
            else:
                (stitched_image, flag) = combine_images(panorama, processed_images[len(processed_images)//2 + i], 'vid', projection)
                if flag==1:
                    g_flag = 1

            panorama = stitched_image.copy()
            # show the panorama
            # cv2.imshow("Panorama", panorama)
            # cv2.waitKey(0)

          
    return (panorama, g_flag)




def video_panorama(video_path):
    cap = cv2.VideoCapture(video_path)
    image_list = []

    # capture n equally spaced frames from the video
    n = 6
    for i in range(n):
        cap.set(1, i * cap.get(7) / n)
        ret, frame = cap.read()
        image_list.append(frame)

    # add the last frame of the video
    cap.set(1, cap.get(7) - 1)
    ret, frame = cap.read()
    image_list.append(frame)

    orig_image_list = image_list.copy()


    (panorama, flag) = stitch_images(image_list, 'vid', 'cyl')
    i = 0
    while flag == 1 and i<7:
        print("Retrying ")
        global focal_length
        focal_length += 100
        # print the new
        print("New Focal Length:", focal_length)
        image_list = orig_image_list.copy()
        (panorama, flag) = stitch_images(image_list, 'vid', 'cyl')
        
        i+=1

                                     
    return panorama








# dir_names = ['Office', 'Mountain']
# for dir_name in dir_names:


#     # Testing Image Stitching
#     image_list = []
#     # read input images from 'data/Images/Mountain' directory
#     for i in range(1, 7):
#         image = cv2.imread(f"data/Images/{dir_name}/{i}.jpg")
#         image_list.append(image)

#     # reverse the image list
#     # image_list = image_list[::-1]
#     panorama = stitch_images(image_list, 'image','cyl')
        


#     # convert the panorama to a linear image

#     # save the final panorama image     
#     cv2.imwrite(f"{dir_name}_panorama.jpg", panorama)

#     # Display result
#     cv2.imshow("Final Panorama", panorama)
#     # cv2.waitKey(0)
#     cv2.destroyAllWindows()



# # Testing Video Stitching
# video_path = "data/Videos/vid2.mp4"
# panorama = video_panorama(video_path)

# apply a perspective transform to align the left and right boundary of the final panorama


# # show the panorama
# cv2.imshow("Final Panorama", panorama)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#  FINAL CODE
    
import sys
import os

def task_part1(input_dir, output_path):
    # Task 1: Processing images in the input directory
    
    print("Processing images in directory:", input_dir)
    
    image_list = []
    # read the input images from the input directory in sorted order
    for file in sorted(os.listdir(input_dir)):
        # print the file name
        print("Reading file:", file)
        image = cv2.imread(os.path.join(input_dir, file))
        image_list.append(image)

    orig_image_list = image_list.copy()

    # stitch the images together to create a panoramic image
    (panorama, flag) = stitch_images(image_list, 'image', 'cyl')
    i = 0
    while flag == 1 and i<7:
        print("Retrying ")
        global focal_length
        focal_length += 100
        print("New Focal Length:", focal_length)
        image_list = orig_image_list.copy()
        (panorama, flag) = stitch_images(image_list, 'image', 'cyl')
        
        i+=1



    # save the final panorama image
    cv2.imwrite(output_path, panorama)

    
    print("Saving panoramic image to:", output_path)

def task_part2(video_path, output_path):
    # Task 2: Processing the video file
    # Your code for processing the video goes here
    print("Processing video file:", video_path)
    
    
    panorama = video_panorama(video_path)
    
    
    # save the final panorama image
    cv2.imwrite(output_path, panorama)

    print("Saving panoramic image to:", output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <part id> <input path> <output path>")
        sys.exit(1)

    part_id = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    global focal_length
    focal_length = 900

    # start time
    start = cv2.getTickCount()


    if part_id == 1:
        task_part1(input_path, output_path)
    elif part_id == 2:
        task_part2(input_path, output_path)
    else:
        print("Invalid part id. Please provide either 1 or 2.")

    # end time
    end = cv2.getTickCount()
    # calculate the time
    time = (end - start) / cv2.getTickFrequency()
    print("Time:", time, "s")













# def laplacian_blending(wi_cpy, image2_cpy):
#        # Read ovewritten images again.
#     imgr = image2_cpy
#     imgl = wi_cpy
#     # Add buffer pixels to right image to make it have equal size to the left one
#     imgr = cv2.copyMakeBorder(imgr, 0, 0, (-imgr.shape[1] + imgl.shape[1]) , 0, cv2.BORDER_CONSTANT,value=255)
#     # imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)

#     #Determine Shape and create un-blended image before blending
#     imgl = imgl[21:, 215:]
#     imgr = imgr[21:, 215:]

#     # parameters
#     g_size = 255  # gaussian mask blur size
#     depth = 3  # pyramid depth size

#     A = imgl.copy()
#     B = imgr.copy()

#     """ PYRAMID BLENDING """
#     row, col, dpt = A.shape
#     # generate Gaussian pyramid for imgA
#     G = A.copy()
#     gp1 = [G]
#     for i in range(depth):
#         G = cv2.pyrDown(G)
#         gp1.append(G)

#     # generate Gaussian pyramid for imgB
#     G = B.copy()
#     gp2 = [G]
#     for i in range(depth):
#         G = cv2.pyrDown(G)
#         gp2.append(G)

#     # generate Laplacian pyramid for imgA
#     lp1 = [gp1[depth-1]]  # store last gaussian image for top of lp pyramid
#     for i in range(depth-1,0,-1):
#         # upsample lower pyramid
#         hr = cv2.pyrUp(gp1[i], dstsize=(gp1[i-1].shape[1], gp1[i-1].shape[0]))
#         lp = cv2.subtract(gp1[i-1], hr, cv2.CV_32F)  # subtract different levels
#         lp1.append(lp)

#     # generate Laplacian pyramid for imgB
#     lp2 = [gp2[depth-1]]  # store last gaussian image for top of lp pyramid
#     for i in range(depth-1,0,-1):
#         # upsample lower pyramid
#         hr = cv2.pyrUp(gp2[i], dstsize=(gp2[i-1].shape[1], gp2[i-1].shape[0]))
#         lp = cv2.subtract(gp2[i-1], hr, cv2.CV_32F)  # subtract different levels
#         lp2.append(lp)

#     # add left and right halves of images in each level
#     LP = []
#     for la, lb in zip(lp1, lp2):
#         row, cols, dpt = la.shape
#         # stack columns (half and half)
#         lp = np.hstack((la[:, :round(cols/2)], lb[:, round(cols/2):]))
#         LP.append(lp)

#     # build Gaussian pyramid from selected region
#     mask = np.zeros((A.shape[1], A.shape[0]))
#     # create mask
#     mask[:, round(A.shape[1]/2):] = 1
#     # blur mask
#     mask = cv2.GaussianBlur(mask, (g_size, g_size), 0)
#     # generate Gaussian pyramid
#     G = mask.copy()
#     GR = [G]
#     for i in range(depth):
#         G = cv2.pyrDown(G)
#         GR.append(G)

#     # add left and right halves of images in each level using Gaussian mask
#     LP_mask = []
#     i = 1
#     for la, lb in zip(lp1, lp2):
#         idx = depth - i

#         # Now blend
#         lp = (1-GR[idx].reshape(GR[idx].shape[1],GR[idx].shape[0],1)) * la + \
#             GR[idx].reshape(GR[idx].shape[1],GR[idx].shape[0],1) * lb
#         lp = np.uint8(lp)  # convert back to uint8
#         LP_mask.append(lp)
#         i += 1

#     # now reconstruct
#     rs = LP[0]
#     for i in range(1, depth):
#         rs = cv2.pyrUp(rs, dstsize=(LP_mask[i].shape[1], LP_mask[i].shape[0]))  # upsample current lp image
#         rs = cv2.add(rs, LP[i])  # add lp image

#     # now Gaussian mask reconstruct
#     rs_mask = LP_mask[0]
#     for i in range(1, depth):
#         rs_mask = cv2.pyrUp(rs_mask,  dstsize=(LP_mask[i].shape[1], LP_mask[i].shape[0]))  # upsample current lp image
#         rs_mask = cv2.add(rs_mask, LP_mask[i])  # add lp image

#     # # remove the black areas  in panaorama
#     # gray = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
#     # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # x, y, w, h = cv2.boundingRect(contours[0])
#     # rs = rs[y:y+h, x:x+w]



#     # display results
#     cv2.imshow('Blended Image', rs)
#     cv2.waitKey(0)
#     return rs