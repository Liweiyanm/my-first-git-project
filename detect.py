#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d
import numpy as np
import time

if __name__ == '__main__':
    file_path = './data/neuvsnap_routine_20220818_151422.pcd'
    #file_path = './data/neuvsnap_routine_20220818_151422.pcd'
    #file_path = './data/neuvsnap_routine_20220818_152143.pcd'
    #60米
    #file_path = './data/neuvsnap_yaw_20220825_154429.pcd'

    pcd = o3d.io.read_point_cloud(file_path)

    #整体包围框
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    #过滤y方向点云，获取1.5米高以下点云（去除天花板）
    y_threshold = 1.5
    points = np.asarray(pcd.points)
    pcd = pcd.select_by_index(np.where(points[:, 1] < y_threshold)[0])

    #pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)

    #有点云数据才处理
    if np.size(np.asarray(pcd.points)) > 0:
        begin_time = time.time()
        #eps：邻居点距离（米）
        #labels返回聚类成功的类别，-1表示没有分到任何类中的点
        labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=3, print_progress=True))
        end_time = time.time()
        runtime = end_time - begin_time
        print(runtime)

        #最大值相当于共有多少个类别
        max_label = np.max(labels)
        print(max(labels))

        min_label = np.min(labels)
        print(min(labels))

        #生成n+1个类别的颜色，n表示聚类成功的类别，1表示没有分类成功的类别
        colors = np.random.randint(255, size=(max_label+1, 3))/255.
        colors = colors[labels]
        #没有分类成功的点设置为黑色
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        box_list = []

        #label是-1的不作为对象，从0开始
        for label in range(0, max_label + 1):
            label_index = np.where(labels == label)
            label_index = np.where(labels == label)  # 提取分类为label的聚类点云下标
            label_pcd = pcd.select_by_index(np.array(label_index)[0])  # 根据下标提取点云点
            #每个类的包围框
            aabb_c = label_pcd.get_axis_aligned_bounding_box()
            aabb_c.color = (0, 1, 0)
            box_list.append(aabb_c)

            #计算中心点位置
            p = np.asarray(label_pcd.points)
            x_avg = np.average(p[:, 0])
            y_avg = np.average(p[:, 1])
            z_avg = np.average(p[:, 2])
            print(z_avg)

            # 点云显示
        o3d.visualization.draw_geometries([pcd,aabb] + box_list, #点云列表
                                          window_name="DBSCAN聚类",
                                          point_show_normal=True,
                                          mesh_show_wireframe=True,
                                          width=2000,  # 窗口宽度
                                          height=2000)  # 窗口高度

