# coding:utf-8

import xml.dom.minidom
from xml.dom import minidom
import xml.etree.ElementTree as ET
import re
import codecs
import numpy as np
import logging


#{
#     "ai_module_name": "auto_login",
#     "dom_tree_url": "/root/workspace/devicepass-ai/syslogin/test-data/一点金库理财-1.3.9/HTC D10w登录.xml",
#     "screen_shot_url": "/root/workspace/devicepass-ai/syslogin/test-data/一点金库理财-1.3.9/HTC  D10w 登录.jpg"
#
# }

unique_id = 1
class XmlParser:
    def __init__(self):
        self.xml_full_path = None
        self.xml_doc = None
        self.node_list = None
        self.node_dict = None

    def get_xml_doc(self, xml_full_path):
        self.xml_doc = minidom.parse(xml_full_path)

    def get_node_list(self):
        return self.xml_doc.getElementsByTagName('node')
    @staticmethod
    def get_emlements(self, element, xpath):
        return element.findall(xpath)

    @staticmethod
    def get_element_by_xpath(self, root, xpath):

        lst = root.findall(xpath)
        if len(lst) < 1:
            print "xml 格式不对!"
            raise Exception("xml格式不对!")
        else:
            return lst

    @staticmethod
    def get_tree(xml_path):
        tree = ET.parse(xml_path)
        return tree

    def walk_data(self, root_node, level, result_list):
        global unique_id
        temp_list = root_node
        result_list.append(temp_list)
        unique_id += 1

        children_node = root_node.getchildren()
        if len(children_node) == 0:
            return
        for child in children_node:
            self.walk_data(child, level + 1, result_list)
        return

    def get_xml_data(self, file_name):
        level = 1
        result_list = []
        try:
            root = ET.parse(file_name).getroot()
            self.walk_data(root, level, result_list)
        except Exception as e:
            logging.critical('Read xml data FAILURE!!!')
            return None
        return result_list

    def read_from_string(self, xml_string):
        level = 1
        result_list = []
        try:
            root = ET.fromstring(xml_string)
            self.walk_data(root, level, result_list)
        except Exception as e:
            logging.critical('Read xml data FAILURE!!!')
            return None
        return result_list

    def xml2str(self, xml):
        f = open(xml, 'rb')
        str1 = ''
        while True:
            line = f.readline()
            if line == '':
                break
            str1 += line
        f.close()
        return str1

    def differ(self, nodes_md5_map1, nodes_md5_map2):

        differ_map = {}
        # if len(nodes_md5_map1) != len(nodes_md5_map2):
        if cmp(nodes_md5_map1, nodes_md5_map2) == 0:
            return None

        for k2, v2 in nodes_md5_map2.items():
            print k2
            found = False
            for k1, v1 in nodes_md5_map1.items():
                if k1 == k2:
                    print 'has same node'
                    found = True
                    break
            if not found:
                differ_map[k2] = v2

        return differ_map
#######################################################
    def coord_transform(self, bounds):
        coord = bounds.strip('[').strip(']').split('][')
        coord_list = list(coord)

        left_top = coord_list[0].split(',')
        right_down = coord_list[1].split(',')

        left = int(left_top[0])
        upper = int(left_top[1])
        right = int(right_down[0])
        down = int(right_down[1])
        return left, upper, right, down

    def get_edit_text_widgets(self, node_list, edit_widget_key_list):
        edit_text_widgets = []
        for n in node_list:
            class_value = n.get('class')
            if class_value in edit_widget_key_list:
                coord_value = n.get('bounds')
                l, u, r, d = self.coord_transform(coord_value)
                if (r-l) < 10 or (d - u) < 10:
                    continue
                edit_text_widgets.append(n)
        return edit_text_widgets

    def get_btn_widget(self, node_list, btn_widget_class_key_list):
        btn_widgets = []
        for n in node_list:
            class_value = n.get('class')
            coord_value = n.get('bounds')
            if class_value in btn_widget_class_key_list:
                l, u, r, d = self.coord_transform(coord_value)
                if (r-l) < 10 or (d - u) < 10:
                    continue
                btn_widgets.append(n)
        return btn_widgets

    def is_login_ui(self, edit_text_widgets, btn_widgets):
        edit_widget_num = len(edit_text_widgets)
        btn_widget_num = len(btn_widgets)

        if  edit_widget_num > 3:
            return False

        coord_dict = {}
        edit_coord_list = []
        for edit_widget in edit_text_widgets:
            edit_bound_val = edit_widget.get('bounds')
            l, u, r, d = self.coord_transform(edit_bound_val)
            edit_coord_list.append([l, u, r, d])
        coord_dict['edit_text'] = edit_coord_list
        edit_list = coord_dict['edit_text']
        # assert len(edit_list) > 1, "Has no edit widget!!!"
        if len(edit_list) < 1:
            return False

        # e1_x1 = edit_list[0][0]; e1_y1 = edit_list[0][1];
        # e1_x2 = edit_list[0][2]; e1_y2 = edit_list[0][3];
        # e2_x1 = edit_list[1][0]; e2_y1 = edit_list[1][1];
        # e2_x2 = edit_list[1][2]; e2_y2 = edit_list[1][3];
        # e1_h = e1_y2 - e1_y1; e1_w = e1_x2 - e1_x1
        # e2_h = e2_y2 - e2_y1; e2_w = e2_x2 - e2_x1

        btn_coord_list = []
        for btn_widget in btn_widgets:
            btn_bound_val = btn_widget.get('bounds')
            l, u, r, d = self.coord_transform(btn_bound_val)
            btn_coord_list.append([l, u, r, d])
        coord_dict['btn'] = btn_coord_list
        btn_list = coord_dict['btn']
        # assert len(btn_list) >= 1, "Has no button widget!!!"
        if len(btn_list) < 1:
            return False
        b1_x1 = btn_list[0][0]; b1_y1 = btn_list[0][1]
        b1_x2 = btn_list[0][2]; b1_y2 = btn_list[0][3]
        b1_h = b1_y2 - b1_y1; b1_w = b1_x2 - b1_x1

        # if e1_w == e2_w and e1_h == e2_h and \
        #     b1_h > e1_h *.75 and b1_w > e1_w *.75:
        #     return True

        return True

    def check_in_attr_key_list(self, attr_val, attr_key_lst):
        attr_val = attr_val.encode("utf-8")
        attr_val = attr_val.lower()
        attr_val = attr_val.split(' ')
        attr_val = ''.join(attr_val)

        if attr_val is None:
            return False

        for u_s in attr_key_lst:
            # utfs = us.encode("utf-8") # if load config data from file, is OK !!
            if (u_s in attr_val) or (u_s == attr_val):
                # print   attr_val,  " matched ", u_s
                return True
            else:
                # print   attr_val, " not matched ", u_s
                continue
        return False

    def xml_estm_edit(self, edit_text_widgets, text_key_list, rcid_key_list):
        edit_widget_num = len(edit_text_widgets)
        # assert edit_widget_num >= 1
        if edit_widget_num < 1:
            return None

        edit_widgets = []
        for n in edit_text_widgets:
            text_val = n.get('text')
            rc_id_val = n.get('resource-id')

            if self.check_in_attr_key_list(text_val, text_key_list) \
                or self.check_in_attr_key_list(rc_id_val, rcid_key_list):
                edit_widgets.append(n)
                # print "xml_estm_edit:", n.get('bounds')

        return edit_widgets


    def xml_estm_button(self, estm_user_edit, estm_pwd_edit, btn_widgets, key_list):
        # assert estm_user_edit is not None and estm_pwd_edit is not None

        # if estm_user_edit is None:
            # return None

        btn_widget_num = len(btn_widgets)
        # assert btn_widget_num >= 1

        if btn_widget_num < 1:
            return None

        usr_d = None
        if estm_user_edit is not None:
            user_coord = estm_user_edit.get('bounds')
            print "user edit text:", user_coord
            usr_l, usr_u, usr_r, usr_d = self.coord_transform(user_coord)

        pwd_d = None
        if estm_pwd_edit is not None:
            pwd_coord = estm_pwd_edit.get('bounds')
            print "password edit text:", pwd_coord
            pwd_l, pwd_u, pwd_r, pwd_d = self.coord_transform(pwd_coord)

        candid_btn_widgets = []

        opt_btn_widget = btn_widgets[0]
        opt_coord_val = opt_btn_widget.get('bounds')
        opt_btn_l, opt_btn_u, opt_btn_r, opt_btn_d = self.coord_transform(opt_coord_val)

        for n in btn_widgets:

            size = len(candid_btn_widgets)
            text_val = n.get('text')
            rc_id_val = n.get('resource-id')
            content_desc = n.get('content-desc')
            coord_val = n.get('bounds')
            btn_l, btn_u, btn_r, btn_d = self.coord_transform(coord_val)

            # if btn_d > usr_d and btn_d > pwd_d:

            if estm_pwd_edit is not None:
                if btn_d > pwd_d:
                    if self.check_in_attr_key_list(text_val, key_list) \
                            or self.check_in_attr_key_list(rc_id_val, key_list) \
                            or self.check_in_attr_key_list(content_desc, key_list):
                        candid_btn_widgets.append(n)
                    else:
                        continue
                else:
                    continue
            if estm_user_edit is not None and estm_pwd_edit is None:
                if btn_d > usr_d:

                    # opt_btn_widget = n
                    # opt_coord_val = opt_btn_widget.get('bounds')
                    # opt_btn_l, opt_btn_u, opt_btn_r, opt_btn_d = self.coord_transform(opt_coord_val)

                    # if btn_d < opt_btn_d:
                    #     opt_btn_widget = n
                    #     opt_coord_val = opt_btn_widget.get('bounds')
                    #     opt_btn_l, opt_btn_u, opt_btn_r, opt_btn_d = self.coord_transform(opt_coord_val)

                    # candid_btn_widgets.append(n)

                    if self.check_in_attr_key_list(text_val, key_list) \
                            or self.check_in_attr_key_list(rc_id_val, key_list) \
                            or self.check_in_attr_key_list(content_desc, key_list):
                        candid_btn_widgets.append(n)
                    else:
                        continue

                else:
                    continue

        return candid_btn_widgets



    def get_widget_list(self, node_list, attr_name, key_list):
        candit_widget_list = []

        attr_name = attr_name.strip()
        for n in node_list:
            value = n.get(attr_name)
            # if re.match(value, key_list):
            if value in key_list:
                # print("class:", n.attributes['class'].value, "coordinate:", n.attributes['bounds'].value)
                candit_widget_list.append(n)
            else:
                continue
        # print "\n"
        return candit_widget_list

    def check_in_key_lis(self, attr_val, cfg_itm_val_lst):
        attr_val = attr_val.encode("utf-8")
        attr_val = attr_val.split(' ')
        attr_val = ''.join(attr_val)

        if not attr_val.strip():
            return False
        for u_s in cfg_itm_val_lst:
            # utfs = us.encode("utf-8") # if load config data from file, is OK !!
            if (u_s in attr_val) or (u_s == attr_val):
                return True
            else:
                continue
        return False


    def estm_widget(self, candit_wid_list, txt_key_list, rc_id_key_list):
        estm_wid_list = []
        for n in candit_wid_list:
            # attr_class_val = n.get('class')
            attr_bound_val = n.get('bounds')
            attr_text_val = n.get('text')
            attr_rc_id_val = n.get('resource-id')

            if self.check_in_key_lis(attr_text_val, txt_key_list):
                estm_wid_list.append(n)
            else:
                if self.check_in_key_lis(attr_rc_id_val,rc_id_key_list):
                    # print("logon button: ", n.get('bounds'))
                    estm_wid_list.append(n)
                else:
                    continue

        return estm_wid_list