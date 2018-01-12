#coding:utf-8
import os
import cv2
import numpy as np
import uuid
import xml_parser as xp
import devicepassai.common.LogUtils as LogUtils
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium import webdriver
# import PIL.ImageChops as imagechops

work_dir = os.path.abspath('.')
IMAGE_FILE_PATH = work_dir + '/test_images/'
OUT_PATH = '/tmp/'

if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

log = LogUtils.MyLog('/tmp/auto_slider_verification.log')

class AutoVerification():
    TEMPLAT_MATCH_THRESHOLD = 0.8
    MIN_HESSIAN = 400
    def __init__(self):
        self.screen1_color = None
        self.screen_img2 = None
        self.btn_center = None

    def read_image(self):
        self.img1_color = cv2.imread(self.img1_path, cv2.IMREAD_COLOR)
        self.img2_color = cv2.imread(self.img2_path, cv2.IMREAD_COLOR)
        assert self.img1_color is not None and self.img2_color is not None
        # cv2.imshow('img1', self.img1_color)
        # cv2.waitKey(0)
        # cv2.imshow('img2', self.img2_color)
        # cv2.waitKey(0)

        return self.img1_color, self.img2_color

    def handle_exception(self, tl_pt, br_pt, screen_color):
        # rect = (tl_pt[0], tl_pt[1], br_pt[0], br_pt[1])

        left = int(tl_pt[0])
        upper = int(tl_pt[1])
        right = int(br_pt[0])
        down = int(br_pt[1])
        mat_grey = screen_color[upper:down, left:right]
        # mat_grey = self.imgCrop2(self.scene_image_grey, rect)

        surf = cv2.xfeatures2d.SURF_create(self.MIN_HESSIAN)
        (kps, descs) = surf.detectAndCompute(mat_grey, None)

        log.info('[FeatureMatching]: Key points number of mathched object: %s' % (str(len(kps))))
        if (len(kps) < 4):
            return False
        return True

    def feature_match(self, btn_color, screen_color):
        matched_rect_coord = []

        img_display = screen_color.copy()

        result_cols = screen_color.shape[1] - btn_color.shape[1] + 1
        result_rows = screen_color.shape[0] - btn_color.shape[0] + 1

        result = cv2.matchTemplate(screen_color, btn_color, cv2.TM_CCOEFF_NORMED)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)

        threshold = self.TEMPLAT_MATCH_THRESHOLD
        max_match_score = 0.0
        after_filter = []
        temp_x = temp_y = max_match_x = max_match_y = 0
        matched_num = 0

        obj_rows = btn_color.shape[0]
        obj_cols = btn_color.shape[1]

        result_rows = result.shape[0]
        result_cols = result.shape[1]
        if result_rows != 0 and result_cols != 0 and result.data:
            max_match_score = float(result[0][0])

        mapScore2Point = {}
        for i in range(0, result_rows):
            for j in range(0, result_cols):
                match_value = float(result[i][j])
                if match_value >= threshold:
                    cv2.rectangle(img_display, (j, i), (j + obj_cols, i + obj_rows), (0, 255, 0), 1, 8)
                    cv2.putText(img_display, str(match_value), (j, i + 100), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

                    if match_value * 1000000 > max_match_score * 1000000:
                        max_match_score = match_value
                        max_match_x = j
                        max_match_y = i
                    mapScore2Point[match_value] = (j, i)
                    matched_num += 1
        log.info('[FeatureMatching]: total matched num: %s' % (str(matched_num)))

        if matched_num == 0:
            return matched_rect_coord

        listScore2Point2_sortedByVal = sorted(mapScore2Point.iteritems(), key=lambda d: d[1][1])

        i = x = y = 0
        s = temp_s = d = 0.0
        l = len(listScore2Point2_sortedByVal)

        mapScore2Point_filtered = {}
        if l != 0:
            temp_s = listScore2Point2_sortedByVal[0][0]
            temp_x = listScore2Point2_sortedByVal[0][1][0]
            temp_y = listScore2Point2_sortedByVal[0][1][1]

            if (l == 1):
                mapScore2Point_filtered[temp_s] = (temp_x, temp_y)

        mean_obj_w_h = (obj_cols + obj_rows) / 4
        for i in range(1, l):
            x = listScore2Point2_sortedByVal[i][1][0]
            y = listScore2Point2_sortedByVal[i][1][1]
            s = listScore2Point2_sortedByVal[i][0]
            d = (x - temp_x) * (x - temp_x) + (y - temp_y) * (y - temp_y)

            if (s > temp_s):
                if d > mean_obj_w_h * mean_obj_w_h:
                    mapScore2Point_filtered[temp_s] = (temp_x, temp_y)
            else:
                if d > mean_obj_w_h * mean_obj_w_h:
                    mapScore2Point_filtered[temp_s] = (temp_x, temp_y)
                else:
                    continue

            temp_s = listScore2Point2_sortedByVal[i][0]
            temp_x = listScore2Point2_sortedByVal[i][1][0]
            temp_y = listScore2Point2_sortedByVal[i][1][1]

        if len(mapScore2Point_filtered) == 0 and len(listScore2Point2_sortedByVal) != 0:
            log.info('vecfiltered.size() ==0 && vecScore2Point.size() != 0')

            k = listScore2Point2_sortedByVal[0][0]
            x = listScore2Point2_sortedByVal[0][1][0]
            y = listScore2Point2_sortedByVal[0][1][1]
            mapScore2Point_filtered[k] = (x, y)

        for i in range(len(listScore2Point2_sortedByVal)):
            k = listScore2Point2_sortedByVal[i][0]
            x = listScore2Point2_sortedByVal[i][1][0]
            y = listScore2Point2_sortedByVal[i][1][1]
            # print str(x) + " => " + str(x) + ", " + str(y)

        filtered_matched_num = len(mapScore2Point_filtered)
        log.info('Filter matched num: %s' % (str(filtered_matched_num)))

        # cv2.imwrite("templ_match_record.png", img_display);
        tl_pt = (max_match_x, max_match_y)
        rd_pt = (max_match_x + obj_cols, max_match_y + obj_rows)
        log.info('[FeatureMatching]: Matched number is: %s' % (str(filtered_matched_num)))

        if not self.handle_exception(tl_pt, rd_pt, self.screen1_color):
            log.info('FeatureMatching]: Key points too little, template matching NOTHING!')
            filtered_matched_num = 0
        else:
            if filtered_matched_num == 0:
                log.info('[FeatureMatching]: Matched numer = 0, template matching NOTHING!')

        matched_rect_coord.append([tl_pt[0], tl_pt[1]])
        matched_rect_coord.append([rd_pt[0], rd_pt[1]])
        return matched_rect_coord

    def calc_differ(self):
        diff = cv2.subtract(self.img1_color,self.img2_color)
        return diff

    def coord_transform(self, coord):
        # logging.info(type(coord), coord)
        coord = coord.strip('[').strip(']').split('][')
        coord_list = list(coord)
        return coord_list

    def crop(self, screen_img_color, coord):
        left = int(coord[0][0]);    upper = int(coord[0][1])
        right = int(coord[1][0]); down = int(coord[1][1])
        cropped = screen_img_color[upper:down, left:right, :]

        assert cropped is not None
        return cropped

    def calc_circle_coord(self, crop, w, h):
        cimg = cv2.medianBlur(crop, 5)
        img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # [[[ 95.5         87.5         78.16968536]]]
        if w == 720:
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=100, param2=100, minRadius=50, maxRadius=100)
        elif w > 720:
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=100, param2=100, minRadius=80, maxRadius=200)
        assert circles is not None, "Detect cicle button FAILURE!!!"

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255 , 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        log.info('Circle center coord: %s, %s' % (i[0], i[1]))
        # cv2.imshow('detected circles', cimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if len(circles) == 1:
            if len(circles[0]) == 1:
                circle0 = circles[0][0]
                return circle0[0], circle0[1], circle0[2]
            elif len(circles[0]) == 2:
                circle0 = circles[0][0]
                circle1 = circles[0][1]

                if circle0[0] < circle1[0]:
                    return circle0[0], circle0[1], circle0[2]
                else:
                    return circle1[0], circle1[1], circle1[2]

    def verify_coord(self, matched_button_ct_x, puzzle_template_ct_x):
        log.info('Matched button center x: %s, puzzle template center x: %s' % (matched_button_ct_x, puzzle_template_ct_x))
        return True

    def get_webview_roi(self,xml_str):
        x1 = x2 = y1 = y2 = None
        xml_parser = xp.XmlParser()
        # xml_str = xml_parser.xml2str(xml_file)
        node_list1 = xml_parser.read_from_string(xml_str)
        if node_list1 is None:
            log.info('Read xml file FAILURE!!!')

        for n in node_list1:
            if n.get('class') == 'android.webkit.WebView':
                bound_value = n.get('bounds')
                x1, y1, x2, y2 = xml_parser.coord_transform(bound_value)
                break

        return (x1, y1, x2, y2)
    # api1
    def matched_btn_coord(self, screen_img1, xml_str):
        '''计算滑动按钮中心点坐标'''
        self.screen1_color = cv2.imread(screen_img1, cv2.IMREAD_COLOR)
        assert self.screen1_color is not None
        # cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
        # cv2.imshow('img1', self.screen1_color)
        # cv2.waitKey(300)
        # cv2.destroyAllWindows()

        (x1, y1, x2, y2) = self.get_webview_roi(xml_str)


        h = self.screen1_color.shape[0]
        w = self.screen1_color.shape[1]
        # x1 = round(0.094444444 * w)
        # y1 = round(0.60859375 * h)
        # x2 = round(0.295833333 * w)
        # y2 = round(0.708203125 * h)

        btn_coord = [[x1, y1], [x2, y2]]
        btn_clip = self.crop(self.screen1_color, btn_coord)
        # cv2.namedWindow('button', cv2.WINDOW_NORMAL)
        # cv2.imshow('button', btn_clip)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cx, cy, radius = self.calc_circle_coord(btn_clip, w, h)

        matched_rect_coord = self.feature_match(btn_clip, self.screen1_color)
        self.btn_ct_x = matched_rect_coord[0][0] + cx
        self.btn_ct_y = matched_rect_coord[0][1] + cy
        self.btn_radius = radius

        # cv2.circle(self.screen1_color, (self.btn_ct_x, self.btn_ct_y), 5, [0, 0, 255], -1)
        # cv2.imwrite(OUT_PATH + 'btn_ct.jpg', self.screen1_color)

        return (self.btn_ct_x, self.btn_ct_y, self.btn_radius)

    # api2
    def calc_offset_x(self, screen_img1, screen_img2):
        '''计算模板和拼图中心点偏移量'''

        self.screen1_color = cv2.imread(screen_img1, cv2.IMREAD_COLOR)
        assert self.screen1_color is not None
        # cv2.imshow('img1', self.screen1_color)
        # cv2.waitKey(0)

        self.screen2_color = cv2.imread(screen_img2, cv2.IMREAD_COLOR)
        assert self.screen2_color is not None
        # cv2.imshow('img2', self.screen2_color)
        # cv2.waitKey(0)

        # gray1 = cv2.cvtColor(self.screen1_color, cv2.COLOR_BGR2GRAY)
        # gray2 = cv2.cvtColor(self.screen2_color, cv2.COLOR_BGR2GRAY)
        diff_img_color = cv2.subtract(self.screen1_color,self.screen2_color)
        # diff_img_color = cv2.subtract(gray1, gray2)

        diff_img_color = abs(diff_img_color)
        # cv2.imshow("diff1", diff_img_color)
        # cv2.waitKey(0)
        cv2.imwrite(OUT_PATH + 'diff.jpg', diff_img_color)

        # diff_img_color = cv2.subtract(self.screen2_color, diff_img_color)
        # cv2.imshow("diff2", diff_img_color)
        # cv2.waitKey(0)
        #
        # diff_img_color = cv2.subtract(self.screen1_color, diff_img_color)
        # # diff_img_color = cv2.subtract(gray1, gray2)
        # cv2.imshow("diff3", diff_img_color)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(diff_img_color, cv2.COLOR_BGR2GRAY)

        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("binary", binary)
        # cv2.waitKey(0)

        median = cv2.medianBlur(gray, 5)
        # median = cv2.medianBlur(diff_img_color, 5)
        edged = cv2.Canny(median, 10, 250)
        # edged = cv2.Canny(binary, 10, 250)
        # cv2.imshow("Edged", edged)
        # cv2.waitKey(0)
        # cv2.imwrite(OUT_PATH + 'Edged.jpg', edged)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # opened = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)

        # find contours (i.e. the 'outlines') in the image and initialize the
        # total number of rectangle found
        _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0

        # assert len(contours) == 2
        coords = []
        # loop over the contours
        for c in contours:
            # approximate the contour
            area = cv2.contourArea(c)
            if area < 100:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(c)
            diff_img_color = cv2.rectangle(diff_img_color, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # if the approximated contour has four points, then assume that the
            # contour is a rectangle and thus has four vertices
            # if len(approx) == 4:

            cv2.drawContours(diff_img_color, [approx], -1, (0, 255, 0), 4)

            M = cv2.moments(c)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            coords.append([cx, cy])

            cv2.circle(diff_img_color, (cx, cy), 5, [0, 0, 255], -1)

            total += 1

        log.info('******FOUND %s  polynomal in that image******' % (total))
        log.info('Polynomal center coords: %s' % (coords))

        out_image_name = str(uuid.uuid4()) + ".jpg"
        # cv2.imshow("Output", diff_img_color)
        # cv2.waitKey(0)
        cv2.imwrite(OUT_PATH + out_image_name, diff_img_color)

        assert self.verify_coord(self.btn_ct_x, coords[0][0])
        offset_x = abs(coords[1][0] - coords[0][0])
        return offset_x

    def fetch_left_non_black_pixel_pos(self, img):
        img_clone = img
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                v = img[j][i]
                if v != 0:
                    img[j][i] = 255
                    # return (j,i)
        return img_clone

    def roi_region_right(self, diff_img_color,binary, btn_cx, btn_radius):
        '''依比例只在随机背景图片区域检测, (440/1440, 847/2560  1288/1440, 1548/2560)'''

        h = binary.shape[0]
        w = binary.shape[1]

        # roi_x1 = btn_cx + btn_radius
        roi_x1 = int(round(w * 0.291666667))
        roi_y1 = int(round(h * 0.330859375))
        roi_x2 = int(round(w * 0.894444444))
        roi_y2 = int(round(h * 0.6046875))

        # roi_binary = binary[roi_y1:roi_y2, roi_x1:roi_x2]
        cv2.rectangle(diff_img_color, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)

        left_point = ()
        right_point =()
        found = False
        for i in range(roi_x1, roi_x2+1):
            for j in range(roi_y1, roi_y2+1):
                v1 = binary[j][i]
                if v1 == 255:
                    left_point = (i, j)
                    found = True
                    break
            if found:
                break

        found = False
        for k in range(roi_x2, roi_x1+1, -1):
            for n in range(roi_y1, roi_y2+1):
                v2 = binary[n][k]
                if v2 == 255:
                    right_point=(k, n)
                    found = True
                    break
            if found:
                break

        if len(left_point) == 0 or len(right_point) == 0:
            return left_point, right_point

        cv2.circle(diff_img_color, left_point, 5, [0, 0, 255], -1)
        cv2.circle(diff_img_color, right_point, 5, [0, 255, 0], -1)
        # cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        # cv2.imshow('ROI', diff_img_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        finale_result_img = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(OUT_PATH + finale_result_img, diff_img_color)

        return left_point, right_point

    def roi_region_left(self, diff_img_color,binary, btn_cx, btn_radius):
        '''依比例只在随机背景图片区域检测, (440/1440, 847/2560  1288/1440, 1548/2560)'''

        h = binary.shape[0]
        w = binary.shape[1]

        # roi_x1 = btn_cx + btn_radius
        roi_x1 = 0
        roi_y1 = int(round(h * 0.330859375))
        roi_x2 = int(round(w * 0.291666667))
        roi_y2 = int(round(h * 0.6046875))

        # roi_binary = binary[roi_y1:roi_y2, roi_x1:roi_x2]
        cv2.rectangle(diff_img_color, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)

        left_point = ()
        right_point =()
        found = False
        for i in range(roi_x1, roi_x2+1):
            for j in range(roi_y1, roi_y2+1):
                v1 = binary[j][i]
                if v1 == 255:
                    left_point = (i, j)
                    found = True
                    break
            if found:
                break

        found = False
        for k in range(roi_x2, roi_x1+1, -1):
            for n in range(roi_y1, roi_y2+1):
                v2 = binary[n][k]
                if v2 == 255:
                    right_point=(k, n)
                    found = True
                    break
            if found:
                break

        if len(left_point) == 0 or len(right_point) == 0:
            return left_point, right_point

        cv2.circle(diff_img_color, left_point, 5, [0, 0, 255], -1)
        cv2.circle(diff_img_color, right_point, 5, [0, 255, 0], -1)
        # cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        # cv2.imshow('ROI', diff_img_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        finale_result_img = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(OUT_PATH + finale_result_img, diff_img_color)

        return left_point, right_point

    def calc_offset_x_ext(self, screen_img1, screen_img2, circle_btn_data):
        '''计算模板和拼图左边缘点坐标差'''
        self.screen1_color = cv2.imread(screen_img1, cv2.IMREAD_COLOR)
        assert self.screen1_color is not None
        # cv2.imshow('img1', self.screen1_color)
        # cv2.waitKey(0)

        self.screen2_color = cv2.imread(screen_img2, cv2.IMREAD_COLOR)
        assert self.screen2_color is not None
        # cv2.imshow('img2', self.screen2_color)
        # cv2.waitKey(0)

        diff_img_color = cv2.subtract(self.screen1_color, self.screen2_color)
        # cv2.imshow("diff1", diff_img_color)
        # cv2.waitKey(0)
        # cv2.imwrite(OUT_PATH + 'diff.jpg', diff_img_color)

        gray = cv2.cvtColor(diff_img_color, cv2.COLOR_BGR2GRAY)

        # gray_clone = self.fetch_left_non_black_pixel_pos(gray)
        # cv2.imshow("gray clone", gray_clone)
        # cv2.waitKey(10)
        # cv2.imwrite('gray_clone.jpg', gray_clone)

        median = cv2.medianBlur(gray, 5)
        edged = cv2.Canny(median, 10, 250)

        # cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
        # cv2.imshow("Edged", edged)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(OUT_PATH + 'edged.jpg', edged)

        binary_name = str(uuid.uuid4()) + '.jpg'
        ret, binary = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(OUT_PATH + binary_name, binary)

        l_left_pt, l_right_pt = self.roi_region_left(diff_img_color, binary, circle_btn_data[0], circle_btn_data[2])
        r_left_pt, r_right_pt = self.roi_region_right(diff_img_color, binary, circle_btn_data[0], circle_btn_data[2])

        if len(l_left_pt) == 0 or len(l_right_pt) == 0 or len(r_left_pt) == 0 or len(r_right_pt) == 0:
            log.error('Calculate edget points FAILURE!!!')
            return ()

        template_ct_x = (l_right_pt[0] + l_left_pt[0]) / (float)(2)
        puzzle_ct_x = (r_right_pt[0] + r_left_pt[0]) / (float)(2)

        offset_x = puzzle_ct_x - template_ct_x

        start_point = (template_ct_x, circle_btn_data[1])

        cv2.circle(self.screen2_color, (int(round(circle_btn_data[0])), int(round(circle_btn_data[1]))), 5, [0, 0, 255], -1)
        cv2.circle(self.screen2_color, (int(round(r_left_pt[0])),  int(round(r_left_pt[1]))), 5, [0, 0, 255], -1)
        cv2.circle(self.screen2_color, (int(round(r_right_pt[0])), int(round(r_right_pt[1]))), 5, [0, 255, 0], -1)
        cv2.circle(self.screen2_color, (int(round(l_left_pt[0])), int(round(l_left_pt[1]))), 5, [0, 0, 255], -1)
        cv2.circle(self.screen2_color, (int(round(l_right_pt[0])), int(round(l_right_pt[1]))), 5, [0, 255, 0], -1)
        cv2.line(self.screen2_color, (int(round(circle_btn_data[0])), int(round(circle_btn_data[1]))), (int(round(puzzle_ct_x)), int(round(circle_btn_data[1]))), [0, 255, 0], 2)
        cv2.line(self.screen2_color, (int(round(puzzle_ct_x)), int(round(circle_btn_data[1]))), (int(round(puzzle_ct_x)), 0), [0, 0, 255], 2)
        cv2.line(self.screen2_color, (int(round(circle_btn_data[0])), int(round(circle_btn_data[1]))),
                 (int(round(circle_btn_data[0])), 0), [255, 0, 0], 2)

        cv2.line(self.screen2_color, (int(round(template_ct_x)), int(round(circle_btn_data[1]))),
                 (int(round(template_ct_x)), 0), [0, 255, 255], 2)

        # cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        # cv2.imshow('ROI', self.screen2_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        finale_result_img = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(OUT_PATH + finale_result_img, self.screen2_color)

        #
        # # edged_clone = self.fetch_left_non_black_pixel_pos(edged)
        # # cv2.imshow("edged clone", edged_clone)
        # # cv2.waitKey(10)
        # # cv2.imwrite('edged_clone.jpg', edged_clone)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # closed = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel)
        # # cv2.imshow("Closed", closed)
        # # cv2.waitKey(0)
        #
        # # find contours (i.e. the 'outlines') in the image and initialize the
        # # total number of rectangle found
        # _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # total = 0
        # coords = []
        # # loop over the contours
        # for c in contours:
        #     # approximate the contour
        #     area = cv2.contourArea(c)
        #     log.info('Contour area: %s' % area)
        #     if area < 100:
        #         log.info('Filter small contour: area = %s.' % area)
        #         continue
        #
        #     peri = cv2.arcLength(c, True)
        #     log.info('Contour arc length: %s' % peri)
        #
        #     # approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #     # x, y, w, h = cv2.boundingRect(c)
        #     # diff_img_color = cv2.rectangle(diff_img_color, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #     cv2.drawContours(diff_img_color, c, -1, (0, 255, 0), 4)
        #
        #     M = cv2.moments(c)
        #
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
        #
        #     coords.append([cx, cy])
        #
        #     cv2.circle(diff_img_color, (cx, cy), 5, [0, 0, 255], -1)
        #     cv2.putText(diff_img_color, str(total), (cx + 30, cy + 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
        #
        #     total += 1
        #
        # log.info('*****FOUND %s  polynomal in that image******' % (total))
        # log.info('Polynomal center coords: %s' % (coords))
        #
        # img1_base_name =  os.path.basename(screen_img1)
        # img2_base_name = os.path.basename(screen_img2)
        # out_image_name = img1_base_name +'_'+img2_base_name + ".jpg"
        # # cv2.imshow("Output", diff_img_color)
        # # cv2.waitKey(0)
        # cv2.imwrite(OUT_PATH + out_image_name, diff_img_color)
        #
        #
        # log.info('Output image name: %s' % out_image_name)
        #
        # log.info('Number of coords: %s' % len(coords))
        # assert self.verify_coord(self.btn_ct_x, coords[0][0])
        # offset_x = abs(coords[1][0] - coords[0][0])
        return (start_point, offset_x)

    # def get_difference(self, image1, image2):
    #     diff = imagechops.difference(image1, image2)
    #     diff.show()
    #     print diff.getbbox()


import csv


if __name__ == '__main__':

    # test_img = IMAGE_FILE_PATH + 'test.png'
    # color = cv2.imread(test_img, cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # fileWriteObj = open(IMAGE_FILE_PATH + 'output.csv', 'w')
    # # for i in range(binary.shape[0]):
    # #     for j in range(binary.shape[1]):
    # #         v = binary[i][j]
    # #         fileWriteObj.write(str(v) + ',')
    # #     fileWriteObj.write('\n')
    # # fileWriteObj.close()
    #
    # left_pt =()
    # right_pt = ()
    # found = False
    # h = binary.shape[0]
    # w = binary.shape[1]
    # for i in range(0,w):
    #     for j in range(0, h):
    #         v = binary[j][i]
    #         if v == 255:
    #             left_pt = (i,j)
    #             found = True
    #             break
    #     if found:
    #         break
    #
    # found = False
    # for i in range(w-1, -1, -1):
    #     for j in range(h):
    #         v = binary[j][i]
    #         if v == 255:
    #             right_pt = (i, j)
    #             found = True
    #             break
    #     if found:
    #         break
    #
    # cv2.circle(color, left_pt, 1, [0, 0, 255], -1)
    # cv2.circle(color, right_pt, 2, [0, 255, 0], -1)
    # cv2.imwrite(OUT_PATH + 'test_final.jpg', color)


#####################################################################################
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    filename_map_list = []
    # filename_map_list.append({'img1': '1_1.png', 'xml1': '1_1.xml', 'img2': '1_2.png'})
    # filename_map_list.append({'img1': '2_1.png', 'xml1': '2_1.xml', 'img2': '2_2.png'})
    filename_map_list.append({'img1': '3_1.png', 'xml1': '3_1.xml', 'img2': '3_2.png'})
    filename_map_list.append({'img1': '4_1.png', 'xml1': '4_1.xml', 'img2': '4_2.png'})
    filename_map_list.append({'img1': '5_1.png', 'xml1': '5_1.xml', 'img2': '5_2.png'})
    filename_map_list.append({'img1': '6_1.png', 'xml1': '6_1.xml', 'img2': '6_2.png'})
    filename_map_list.append({'img1': '7_1.png', 'xml1': '7_1.xml', 'img2': '7_2.png'})
    filename_map_list.append({'img1': '8_1.png', 'xml1': '8_1.xml', 'img2': '8_2.png'})
    filename_map_list.append({'img1': '9_1.png', 'xml1': '9_1.xml', 'img2': '9_2.png'})
    filename_map_list.append({'img1': '10_1.png', 'xml1': '10_1.xml', 'img2': '10_2.png'})

    auto_verify = AutoVerification()

    # csvfile = file(OUT_PATH + 'test_result.csv', 'wb')
    # writer = csv.writer(csvfile)
    # writer.writerow(['文件名', '返回值', '结果图像', '备注'])
    i = 0
    xml_str = None
    for n in filename_map_list:
        log.info('------------------screen shot image: '+ str(i+1) + '----------------------------------')
        img1_color = IMAGE_FILE_PATH + n['img1']

        # btn_coord = [[113,1170],[290,1340]]

        # 计算滑动按钮中心点坐标和圆半径
        # 如果第2个参数xml修改为字符串方式，调用时修改为字符串方式，否则出错
        # circle_btn_data = auto_verify.matched_btn_coord(img1_color, IMAGE_FILE_PATH + n['xml1'])
        circle_btn_data = auto_verify.matched_btn_coord(img1_color, xml_str)

        # api2
        img2_color = IMAGE_FILE_PATH + n['img2']

        # auto_verify.get_difference(img1_color, img2_color)

        # 计算模板和拼图中心点偏移量
        result  = auto_verify.calc_offset_x_ext(img1_color, img2_color, circle_btn_data)
        if len(result) == 0:
            log.error('Calculate edges points FAILURE!!!')
            continue
        start_point = result[0]
        offset_x = result[1]
        log.info('-----------------Start point: (%s,%s)------------------------' % (str(start_point[0]), str(start_point[1])))
        log.info('-----------------Puzzle between template offset x: %s------------------------' % (str(offset_x)))

