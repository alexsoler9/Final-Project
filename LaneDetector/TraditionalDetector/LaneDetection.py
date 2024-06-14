import cv2 
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection:
    def __init__(self):
        self.right_slope = []
        self.left_slope = []
        self.right_intercept = []
        self.left_intercept = []
    
    def color_threshold(self, image, mask=None, dilate=False, debug=False):
        # Apply mask if provided
        # Apply mask if provided
        if mask is not None:
            filtered_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            filtered_image = image.copy()

            # Define image to HLS color space
        hls_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HLS)

        # Define white and yellow color ranges in HLS space
        lower_white = np.array([0, 100, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        lower_yelow = np.array([10, 0, 100], dtype=np.uint8)
        upper_yelow = np.array([40, 255, 255], dtype=np.uint8)

        # Threshold image to detect white and yellow colors
        white_mask = cv2.inRange(hls_image, lower_white, upper_white)
        yellow_mask = cv2.inRange(hls_image, lower_yelow, upper_yelow)

        # Combine white and yellow masks
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(color_mask, 50, 255, cv2.THRESH_BINARY)
        
        if dilate:
            kernel = np.ones((10, 10), np.uint8)  # square kernel
            thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=1)

        # Debugging: Display intermediate images
        if debug:
            self.display_images([filtered_image, hls_image, white_mask, yellow_mask, color_mask, thresholded_image],
                                ['Filtered Image', 'HLS Image', 'White Mask', 'Yellow Mask', 'Combined Mask', 'Thresholded Image'])
            # Compute the number of white pixels
            num_white_pixels = cv2.countNonZero(mask)
            print("Number of white pixels (entry):", num_white_pixels)
            num_white_pixels = cv2.countNonZero(thresholded_image)
            print("Number of white pixels (entry):", num_white_pixels)
        return thresholded_image
        

    def sobel_threshold(self, image, sobel_threshold=50, debug=False):
        """
        Process an input image: converts to grayscale, applies Gaussian blur,
        applies Sobel filtering in the diagonal direction, thresholds the Sobel image, 
        and dilates the thresholded image.
        
        Args: 
            image (numpy.ndarray): Input image.
            sobel_threshold (int, optional): Threshold value for Sobel filtering.
            debug (bool, optional): Whether to display intermediate images for debugging.
            
        Returns:
            numpy.ndarray: Thresholded image in grayscale format
        """
        # Convert image to grayscale if not in grayscale
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Apply a median blur
        blurred_image = cv2.medianBlur(gray_image, 31)
        #blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)
        # Apply Sobel
        sobelxy = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y

        #Threshold the Sobel image
        _, thresholded_image = cv2.threshold(sobelxy, sobel_threshold, 255, cv2.THRESH_BINARY)
        # Dilate the thresholded image
        #kernel = np.ones((10, 10), np.uint8)  # square kernel
        #img_th = cv2.dilate(thresholded_image, kernel, iterations=1)
        img_th = thresholded_image.astype(np.uint8)
        
        if debug:
            self.display_images([blurred_image, sobelxy, thresholded_image],
                                ['Blurred Image', 'Sobel Image', 'Thresholded Image'])

        return img_th

    def region_of_interest(self, image, type="default", vertices=None):
        x = int(image.shape[1])
        y = int(image.shape[0])
        if type == "default":
            shape = np.array([[int(0), int(0.7*y)],
                                [int(x), int(0.7*y)], 
                                [int(x), int(0.4*y)], 
                                [int(0), int(0.4*y)]])
        elif type == "lane":
            shape = np.array([[int(0), int(0.7*y)],
                        [int(x), int(0.7*y)],
                        [int(x), int(0.6*y)], 
                        [int(0.7*x), int(0.35*y)],
                        [int(0.3*x), int(0.35*y)],
                        [int(0), int(0.5*y)]])
        #define numpy array with the dimensions of image
        mask = np.zeros_like(image)

        #uses 3 channels or 1 channel for color depending on input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #create a polygon with the mask color
        cv2.fillPoly(mask, np.int32([shape]),ignore_mask_color)
        
        #return the image only where the mask pixels are not zero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, 
                                maxLineGap = max_line_gap)
        line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype= np.uint8)
        polygon = self.plot_lines(line_img, lines)
        return line_img, polygon
        

    def plot_lines(self, image, lines, thickness=5, debug=False):
        right_color = [0, 255, 0]
        left_color = [255, 0, 0]
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 != x2:
                        slope = (y1-y2)/(x1-x2)
                        angle = np.math.atan(slope)*180./np.pi
                        print(angle)
                        if 30 < angle < 75:
                            if x1 > 500:
                                yintercept = y2 - (slope*x2)
                                self.right_slope.append(slope)
                                self.right_intercept.append(yintercept)

                            else: None
                        elif -75 < angle < -30:
                            if x1 < 600:
                                yintercept = y2 - (slope*x2)
                                self.left_slope.append(slope)
                                self.left_intercept.append(yintercept)
                            else: None
                    else:
                        if debug:
                            print(f"x1 {x1} y1 {y1} x2 {x2} y2 {y2}")
        elif lines is None:
            self.left_slope.append(np.mean(self.left_slope[-30:])*0.9)
            self.left_intercept.append(np.mean(self.left_intercept[-30:])*1.5)
            self.right_slope.append(np.mean(self.right_slope[-30:])*1.1)
            self.right_intercept.append(np.mean(self.right_intercept[-30:])*0.5)

        # use slicing to find the average previous frames, which makes lanes less likely to shift
        average = 90

        # Check if the length of the list exceeds three times the average
        if len(self.left_slope) > 3*average:
            self.left_slope = self.left_slope[-average:]
            self.left_intercept = self.left_intercept[-average:]
        if len(self.right_slope) > 3*average:
            self.right_slope = self.right_slope[-average:]
            self.right_intercept = self.right_intercept[-average:]
        
        left_avg_slope = np.mean(self.left_slope[-average:])
        left_avg_intercept = np.mean(self.left_intercept[-average:])

        right_avg_slope = np.mean(self.right_slope[-average:])
        right_avg_intercept = np.mean(self.right_intercept[-average:])
        
        intercept_x = (right_avg_intercept - left_avg_intercept) / (left_avg_slope - right_avg_slope)
        intercept_y = max(left_avg_slope * intercept_x + left_avg_intercept, 0.3*image.shape[0])

        if debug:
            print("left_avg_slope:", left_avg_slope)
            print("right_avg_slope:", left_avg_slope)
            print("left_avg_intercept:", left_avg_intercept)
            print("right_avg_intercept:", right_avg_intercept)
            print("Intercept point (x, y):", (intercept_x, intercept_y))

        try:
            left_line_x1 = int((intercept_y - left_avg_intercept) / left_avg_slope)
            left_line_x2 = int((image.shape[0] - left_avg_intercept) / left_avg_slope) 

            right_line_x1 = int((intercept_y - right_avg_intercept) / right_avg_slope)
            right_line_x2 = int((image.shape[0] - right_avg_intercept) / right_avg_slope)

            pts = np.array([[left_line_x1, int(intercept_y)], [left_line_x2, int(image.shape[0])],
                            [right_line_x2, int(image.shape[0])],[right_line_x1, int(intercept_y)]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], (0, 0, 255))

            cv2.line(image, (left_line_x1, int(intercept_y)), 
                    (left_line_x2, int(image.shape[0])),
                    left_color, 10)
            
            cv2.line(image, (right_line_x1, int(intercept_y)), 
                    (right_line_x2, int(image.shape[0])),
                    right_color, 10)
            return pts.astype(np.int32)
        except ValueError:
            pass
        
    def resize(image, factor=0.5):
        image=cv2.resize(image,None, fx=factor, fy=factor,interpolation=cv2.INTER_AREA)
        return image

    def process_video(self, video_path):
        """
        Process each frame of a video: apply Sobel threshodling and color thresholding,
        displaying the processed frames.
        
        Args:
            video_path (str): Path to the input video file.
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Loop trough each frame of the video
        while cap.isOpened():
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_copy = frame.copy()
            roi = self.region_of_interest(frame_copy)
            roi_test = roi
            # Process the frame using sobel_threshold and color_threshold functions
            sobel_result = self.sobel_threshold(roi_test, sobel_threshold=20)
            #roi = region_of_interest(sobel_result,type="lane")
            result_image = self.color_threshold(frame, mask=sobel_result, dilate=True)
            canny = cv2.Canny(result_image, 50, 120)
            roi = self.region_of_interest(canny, type="lane")
            my_line, _ = self.hough_lines(roi, 2, np.pi/180, 100, 40, 14)
            weighted_img = cv2.addWeighted(my_line, 1, frame, 0.8, 0)

            original=cv2.resize(canny,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=cv2.resize(weighted_img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            cv2.imshow('Original Image',original)
            cv2.imshow('Processed Frame',image)
            

            key = cv2.waitKey(15)
            # If "q" is pressed, break the loop
            if key == ord("q"):
                break
            # If "p" is pressed, pause the video until any key is pressed.
            elif key == ord("p"):
                cv2.waitKey(0)
            elif key == ord("s"):
                self.sobel_threshold(roi_test, sobel_threshold=10, debug=True)
                cv2.waitKey(0)
            elif key == ord("c"):
                self.color_threshold(frame, mask=sobel_result, debug=True)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

    def process_image(self, image):
        pass

    def display_images(self, images , titles=None):
        """
        Helper funtion to display a list of images in subplots
        
        Args:
            images (list of numpy arrays): List of images to display.
            titles (list of str, optional): List of titles for each image
        """
        num_images = len(images)

        # Determmine the number of rows and columns for subplots
        num_rows = int(num_images**0.5)
        num_cols = (num_images//num_rows) + (1 if num_images % num_rows > 0 else 0)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        axes = axes.flatten()

        for i in range(num_images):
            ax = axes[i]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i] if i < len(titles) else f"Image {i+1}")
        
        # Remove any extra empty subplots
        for j in range(num_images, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

