import numpy as np
import timeit
import math


def check_vertex_inside_rectangle_1(a, b):
    result = ((a[:, :, :4, 0] - b[:, :, 1:2, 0]) * b[:, :, 4:5, 0] + (
        a[:, :, :4, 1] - b[:, :, 1:2, 1]) * b[:, :, 4:5, 1] >= 0) & (
                 (a[:, :, :4, 0] - b[:, :, 1:2, 0]) * b[:, :, 5:6, 0] + (
                     a[:, :, :4, 1] - b[:, :, 1:2, 1]) * b[:, :, 5:6, 1] >= 0) & (
                 b[:, :, 4:5, 0] ** 2 + b[:, :, 4:5, 1] ** 2 >= (
                     a[:, :, :4, 0] - b[:, :, 1:2, 0]) * b[:, :, 4:5, 0] + (
                     a[:, :, :4, 1] - b[:, :, 1:2, 1]) * b[:, :, 4:5, 1]) & (
                 b[:, :, 5:6, 0] ** 2 + b[:, :, 5:6, 1] ** 2 >= (
                     a[:, :, :4, 0] - b[:, :, 1:2, 0]) * b[:, :, 5:6, 0] + (
                     a[:, :, :4, 1] - b[:, :, 1:2, 1]) * b[:, :, 5:6, 1])
    return result

def check_vertex_inside_rectangle_2(a, b):
    result = ((b[:, :, :4, 0] - a[:, :, 1:2, 0]) * a[:, :, 4:5, 0] + (
        b[:, :, :4, 1] - a[:, :, 1:2, 1]) * a[:, :, 4:5, 1] >= 0) & (
                 (b[:, :, :4, 0] - a[:, :, 1:2, 0]) * a[:, :, 5:6, 0] + (
                     b[:, :, :4, 1] - a[:, :, 1:2, 1]) * a[:, :, 5:6, 1] >= 0) & (
                 a[:, :, 4:5, 0] ** 2 + a[:, :, 4:5, 1] ** 2 >= (
                     b[:, :, :4, 0] - a[:, :, 1:2, 0]) * a[:, :, 4:5, 0] + (
                     b[:, :, :4, 1] - a[:, :, 1:2, 1]) * a[:, :, 4:5, 1]) & (
                 a[:, :, 5:6, 0] ** 2 + a[:, :, 5:6, 1] ** 2 >= (
                     b[:, :, :4, 0] - a[:, :, 1:2, 0]) * a[:, :, 5:6, 0] + (
                     b[:, :, :4, 1] - a[:, :, 1:2, 1]) * a[:, :, 5:6, 1])

    return result


def check_line_segment_intersection(a, b):
    x1, y1 = a[:, :, 0:1, 0], a[:, :, 0:1, 1]
    x2, y2 = a[:, :, 1:2, 0], a[:, :, 1:2, 1]
    x3, y3 = b[:, :, 0:1, 0], b[:, :, 0:1, 1]
    x4, y4 = b[:, :, 1:2, 0], b[:, :, 1:2, 1]
    ta_1 = (y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)
    ta_2 = (y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)
    ta_3 = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    out = np.zeros_like(ta_1)
    ta = np.divide(ta_1, ta_3, out=out, where=(ta_3 != 0) & (ta_1 / ta_3 > 0) & (ta_1 / ta_3 < 1))
    out = np.zeros_like(ta_2)
    tb = np.divide(ta_2, ta_3, out=out, where=(ta_3 != 0) & (ta_2 / ta_3 > 0) & (ta_2 / ta_3 < 1))
    ta[tb <= 0] = 0
    return ta


def get_point(a, b):
    a1 = a[:, :, 0:1, :]
    a2 = a[:, :, 1:2, :]
    b = np.expand_dims(b, axis=3)
    out = np.zeros((b.shape[0],b.shape[1], 4, 2)) - 1
    result = np.add((a2 - a1) * b, a1, out=out, where=b != 0)
    result[result == 0] = 1e-4
    result[result == -1] = 0
    return result


def Subtract(a, b):
    return np.subtract(a, b, out=np.zeros_like(a), where=a != 0)


def PolyArea(a, b):
    return 0.5 * np.abs(np.sum(np.multiply(a, np.roll(b, 1, axis=2), out=np.zeros_like(a)),
                               axis=2) - np.sum(np.multiply(b, np.roll(a, 1, axis=2), out=np.zeros_like(b)), axis=2))


def intersection_over_union_cpu(bbox1,bbox2):
    if bbox1.ndim == 1:
        bbox1 = bbox1.reshape((1, -1))
    if bbox2.ndim == 1:
        bbox2 = bbox2.reshape((1, -1))
    num_bbox1 = bbox1.shape[0]
    num_bbox2 = bbox2.shape[0]
    # coordinate of list of bbox in order (A -> B -> D -> C)
    a_bbox = np.zeros((num_bbox1 + num_bbox2, 6, 2))

    a_bbox[:num_bbox1, 0, 0] = bbox1[:, 0] - bbox1[:, 2] / 2 * np.cos(bbox1[:, 4]) - bbox1[:,
                                                              3] / 2 * np.sin(
            bbox1[:, 4])  # A_x
    a_bbox[:num_bbox1, 0, 1] = bbox1[:, 1] - bbox1[:, 3] / 2 * np.cos(bbox1[:, 4]) + bbox1[:,
                                                              2] / 2 * np.sin(
            bbox1[:, 4])  # A_y
    a_bbox[:num_bbox1, 1, 0] = bbox1[:, 0] + bbox1[:, 2] / 2 * np.cos(bbox1[:, 4]) - bbox1[:,
                                                              3] / 2 * np.sin(
            bbox1[:, 4])  # B_x
    a_bbox[:num_bbox1, 1, 1] = bbox1[:, 1] - bbox1[:, 3] / 2 * np.cos(bbox1[:, 4]) - bbox1[:,
                                                              2] / 2 * np.sin(
            bbox1[:, 4])  # B_y
    a_bbox[:num_bbox1, 2, 0] = bbox1[:, 0] + bbox1[:, 2] / 2 * np.cos(bbox1[:, 4]) + bbox1[:,
                                                              3] / 2 * np.sin(
            bbox1[:, 4])  # D_x
    a_bbox[:num_bbox1, 2, 1] = bbox1[:, 1] + bbox1[:, 3] / 2 * np.cos(bbox1[:, 4]) - bbox1[:,
                                                              2] / 2 * np.sin(
            bbox1[:, 4])  # D_y
    a_bbox[:num_bbox1, 3, 0] = bbox1[:, 0] - bbox1[:, 2] / 2 * np.cos(bbox1[:, 4]) + bbox1[:,
                                                              3] / 2 * np.sin(
            bbox1[:, 4])  # C_x
    a_bbox[:num_bbox1, 3, 1] = bbox1[:, 1] + bbox1[:, 3] / 2 * np.cos(bbox1[:, 4]) + bbox1[:,
                                                              2] / 2 * np.sin(
            bbox1[:, 4])  # C_y

    a_bbox[num_bbox1:, 0, 0] = bbox2[:, 0] - bbox2[:, 2] / 2 * np.cos(bbox2[:, 4]) - bbox2[:,
                                                              3] / 2 * np.sin(
            bbox2[:, 4])  # A_x
    a_bbox[num_bbox1:, 0, 1] = bbox2[:, 1] - bbox2[:, 3] / 2 * np.cos(bbox2[:, 4]) + bbox2[:,
                                                              2] / 2 * np.sin(
            bbox2[:, 4])  # A_y
    a_bbox[num_bbox1:, 1, 0] = bbox2[:, 0] + bbox2[:, 2] / 2 * np.cos(bbox2[:, 4]) - bbox2[:,
                                                              3] / 2 * np.sin(
            bbox2[:, 4])  # B_x
    a_bbox[num_bbox1:, 1, 1] = bbox2[:, 1] - bbox2[:, 3] / 2 * np.cos(bbox2[:, 4]) - bbox2[:,
                                                              2] / 2 * np.sin(
            bbox2[:, 4])  # B_y
    a_bbox[num_bbox1:, 2, 0] = bbox2[:, 0] + bbox2[:, 2] / 2 * np.cos(bbox2[:, 4]) + bbox2[:,
                                                              3] / 2 * np.sin(
            bbox2[:, 4])  # D_x
    a_bbox[num_bbox1:, 2, 1] = bbox2[:, 1] + bbox2[:, 3] / 2 * np.cos(bbox2[:, 4]) - bbox2[:,
                                                              2] / 2 * np.sin(
            bbox2[:, 4])  # D_y
    a_bbox[num_bbox1:, 3, 0] = bbox2[:, 0] - bbox2[:, 2] / 2 * np.cos(bbox2[:, 4]) + bbox2[:,
                                                              3] / 2 * np.sin(
            bbox2[:, 4])  # C_x
    a_bbox[num_bbox1:, 3, 1] = bbox2[:, 1] + bbox2[:, 3] / 2 * np.cos(bbox2[:, 4]) + bbox2[:,
                                                              2] / 2 * np.sin(
            bbox2[:, 4])  # C_y

    a_bbox[:, 4:5, 0] = a_bbox[:, 0:1, 0] - a_bbox[:, 1:2, 0]  # BA_x
    a_bbox[:, 4:5, 1] = a_bbox[:, 0:1, 1] - a_bbox[:, 1:2, 1]  # BA_y
    a_bbox[:, 5:6, 0] = a_bbox[:, 2:3, 0] - a_bbox[:, 1:2, 0]  # BD_x
    a_bbox[:, 5:6, 1] = a_bbox[:, 2:3, 1] - a_bbox[:, 1:2, 1]  # BD_y

    bbox1 = np.copy(a_bbox[:num_bbox1])
    bbox2 = np.copy(a_bbox[num_bbox1:])
    bbox1 = np.expand_dims(bbox1, axis=1)
    bbox2 = np.expand_dims(bbox2, axis=0)
    a_bbox = np.zeros(
        (num_bbox1, num_bbox2, 24))

    a_bbox[:, :, 0:4] = check_vertex_inside_rectangle_1(bbox1, bbox2)
    a_bbox[:, :, 4:8] = check_vertex_inside_rectangle_2(bbox1, bbox2)
    for i in range(4):
        for j in range(4):
            a_bbox[:, :, 8 + (i * 4) + j:9 + (i * 4) + j] = check_line_segment_intersection(
                bbox1[:, :, [i % 4, (i + 1) % 4], :], bbox2[:, :, [j % 4, (j + 1) % 4], :])

    list_of_vertices = np.zeros((num_bbox1, num_bbox2, 24, 2))
    list_of_vertices[:, :, :4, :] = np.expand_dims(a_bbox[:, :, :4], axis=3) * bbox1[:, :, :4, :]
    list_of_vertices[:, :, 4:8, :] = np.expand_dims(a_bbox[:, :, 4:8], axis=3) * bbox2[:, :, :4, :]
    for i in range(4):
        list_of_vertices[:, :, 8 + i * 4:8 + (i + 1) * 4, :] = get_point(bbox1[:, :, [i % 4, (i + 1) % 4], :],
                                                                         a_bbox[:, :, 8 + i * 4:8 + (i + 1) * 4])
    out = np.zeros((list_of_vertices.shape[0], list_of_vertices.shape[1],2))
    mean = np.divide(list_of_vertices.sum(axis=2), (list_of_vertices != 0).sum(axis=2),where=list_of_vertices.sum(axis=2)!=0, out = out)
    mean_vertices = np.subtract(list_of_vertices, np.expand_dims(mean, axis=2), out = np.zeros_like(list_of_vertices),where=list_of_vertices!=0)
    arctan = np.arctan2(mean_vertices[:, :,:, 1], mean_vertices[:, :, :,0]) + 4
    arctan[(mean_vertices[:,:,:,0]==0) &(mean_vertices[:,:,:,1]==0)] = 0
    arctan = arctan.reshape((-1, arctan.shape[-1]))
    sorted_arc = arctan.argsort()
    list_of_vertices = list_of_vertices.reshape((num_bbox1 * num_bbox2, 24, 2))
    sorted_list = list_of_vertices[np.arange(list_of_vertices.shape[0])[:, np.newaxis], sorted_arc]
    # print sorted_list
    sorted_list = sorted_list.reshape((num_bbox1, num_bbox2, 24, 2))
    sorted_list = Subtract(sorted_list, sorted_list[:, :, -1:, :])[:,:, -8:, :]
    final = PolyArea(sorted_list[:, :, :, 0], sorted_list[:, :, :, 1])
    return final


if __name__ == '__main__':
    # bbox1 = np.array([24, 24, 8, 8, math.pi/4])
    bbox1 = np.array([[22, 22, 8, 8, 44*math.pi / 180], [20, 20, 10, 10, math.pi * 2 / 4], [16, 16, 20, 20, math.pi / 6],
                      [24, 24, 30, 30, 29*math.pi / 180], [40, 40, 8, 8, 0]])
    bbox2 = np.array([[22, 22, 8, 8, math.pi / 4], [20, 20, 10, 10, math.pi * 2 / 4], [16, 16, 20, 20, math.pi / 6],
                  [24, 24, 30, 30, math.pi / 6], [40, 40, 8, 8, 0]])


    start_time = timeit.default_timer()
    areas = intersection_over_union_cpu(bbox1, bbox2)
    end_time = timeit.default_timer()
    print areas
    print "the code run for %s second" % (end_time - start_time)
