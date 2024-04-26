# import necessary libraries
from copy import copy
from pyvis.network import Network
import pm4py
from pm4py.visualization.common.utils import *
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.objects.conversion.log import converter as log_converter
import matplotlib as mpl 
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import warnings
# suppress warnings
warnings.simplefilter("ignore")

# instantiate color maps and normalizer
rb_cm = mpl.cm.get_cmap('seismic')
grey_cm = mpl.cm.get_cmap('coolwarm')
norm = Normalize(vmin=-100, vmax=100)


def divide(numerator, denominator):
    """
    Returns the division of two number, catches division by 0

    Parameters
    -----------
    numerator: float
    denominator: float

    Returns
    ----------
    div_result: float
        result of division
    """
    div_result = numerator / denominator if denominator > 0 else 0
    return div_result

def concat_logs(log_1, log_2, case_id_col='case:concept:name'):
    """
    Concatenates two event logs, reassigns case ids if overlapping

    Parameters
    -----------
    log_1: event log/ DataFrame
        input log 1
    log_2: event log/ DataFrame
        input log 2
    case_id_col: str
        Name of the case identifier column

    Returns
    -----------
    concat_log: DataFrame
        Concatenation of two input logs
    """
    # convert xes event logs to DataFrames
    log1_df = pm4py.convert_to_dataframe(log_1)
    log2_df = pm4py.convert_to_dataframe(log_2)
    # check if case identifier overlap, if so, reassigns numerical id 
    if len(set(list(log1_df[case_id_col].unique()) + list(log2_df[case_id_col].unique()))) != (len(log1_df[case_id_col].unique()) + len(log2_df[case_id_col].unique())):
        new_case_ids1 = np.arange(0, len(log1_df[case_id_col].unique()))
        new_case_ids1 = dict(zip(list(log1_df[case_id_col].unique()), list(new_case_ids1)))
        log1_df[case_id_col] = log1_df[case_id_col].replace(new_case_ids1)
        log1_df[case_id_col] = log1_df[case_id_col].astype(str)
        new_case_ids2 = np.arange(len(log1_df[case_id_col].unique()), len(log2_df[case_id_col].unique())+ len(log1_df[case_id_col].unique()) +1)
        new_case_ids2 = dict(zip(list(log2_df[case_id_col].unique()), list(new_case_ids2)))
        log2_df[case_id_col] = log2_df[case_id_col].replace(new_case_ids2)
        log2_df[case_id_col] = log2_df[case_id_col].astype(str)
    # concatenate DataFrames
    concat_log = pd.concat([log1_df, log2_df])
    return concat_log

def get_case_coverage_edge(log, edge_out, edge_in, num_cases, activity_col_name='concept:name'):
    """
    Determines case coverage of an edge in percentage
    
    Parameters
    -----------
    log: DataFrame
        Event log
    edge_out: str
        Name of activity that is predecessor 
    edge_in: str
        Name of activity that is successor
    num_cases: int
        Number of cases in log
    activity_col_name: str
        Name of activity column
    
    Returns
    -----------
    case_coverage: float, percentage
        Case coverage of edge
    """
    paths_count = 0
    for trace in log:
        for i in range(0, len(trace) - 1):
            if activity_col_name in trace[i] and activity_col_name in trace[i + 1]:
                if (trace[i][activity_col_name] == edge_out) and (trace[i + 1][activity_col_name] == edge_in):
                    paths_count += 1
                    break

    case_coverage = divide(paths_count, num_cases)
    case_coverage = round(case_coverage*100, 1)
    return case_coverage

def color_edge(rel_freq_diff):
    """
    Determines color of edge according to relative frequency difference

    Parameters
    -----------
    value: float
        Relative frequency difference
    
    Returns
    -----------
    color: hex color code
        Color of edge
    """
    percentage= rel_freq_diff*100
    # if < 5%, color grey
    if abs(percentage) < 5:
        rgb = grey_cm(norm(percentage))[:3]
        color = mpl.colors.rgb2hex(rgb)
    else:
        rgb = rb_cm(norm(percentage))[:3]
        color = mpl.colors.rgb2hex(rgb)
    return color

def color_node(rel_cov_diff):
    """
    Determines color of node according to relative case coverage difference

    Parameters
    -----------
    rel_cov_diff: float
        Relative case coverage difference in percentage
    
    Returns
    -----------
    color: hex color code
        Color of node
    """
    rgb = rb_cm(norm(rel_cov_diff))[:3]
    color = mpl.colors.rgb2hex(rgb)
    return color

def node_thickness(abs1, abs2, total):
    """
    Determines thickness of node border according to case coverage

    Parameters
    -----------
    abs1: int
        Case frequency in log 1
    abs2: int
        Case frequency in log 2
    total: int
        Number of cases combined
    
    Returns
    -----------
    thickness: float
        Thickness of node border
    """
    instances_sum = abs1 + abs2 
    rel_freq = divide(instances_sum, total)
    if rel_freq == 0:
        thickness = 0.25
    else:
        thickness = rel_freq*3 + 0.25
    return thickness

def get_rel_freq_diff_vert(cov1, cov2):
    """
    Determines relative frequency difference of case coverage for node

    Parameters
    -----------
    cov1: float, percentage
        Case coverage in log 1
    cov2: float, percentage
        Case coverage in log 2
   
    Returns
    -----------
    rel_freq_diff_vert: float, percentage
        Coverage difference 
    color: hex code 
        Color of node
    """
    rel_freq_diff_vert = cov1 - cov2

    # color of nodee
    color = color_node(rel_freq_diff_vert)

    # relative case coverage difference
    rel_freq_diff_vert = round(rel_freq_diff_vert, 1)

    return rel_freq_diff_vert, color 

def get_outgoing_edges_frequency(node, log_edges):
    """
    get frequencies of outgoing edges from node
    node: str, name of activity
    log_edges: dict, edges and their frequency in log
    returns: 
        freq: int, frequency of edge
    """
    for edge, freq in log_edges.items():
        if node == edge[0]:
            yield freq


def relative_freq_diff_width(edge, L1_edges, L2_edges):
    """
    Determines width of edge

    Parameters
    -----------
    edge: tuple
        (preceeding activity, succeeding activity)
    L1_edges: dict
        Edges and their frequency in log 1
    L2_edges: dict
        Edges and their frequency in log 2
    
    Returns
    -----------
    rel_freq_dif: float
        Relative frequency difference
    width: float
        Width of edge
    freq_L1: int
        Absolute frequency of edge in log 1
    freq_L2: int
        Absolute frequency of edge in log 2
    """

    # frequency for each log
    if edge in L1_edges:
        freq_L1 = L1_edges[edge]
        # total outgoing edge frequency from edge[0]
        freq_sum_L1 = np.sum(list(get_outgoing_edges_frequency(edge[0], L1_edges)))
    else:
        freq_L1 = 0
        freq_sum_L1 = 0
    if edge in L2_edges:
        freq_L2 = L2_edges[edge]
        freq_sum_L2 = np.sum(list(get_outgoing_edges_frequency(edge[0], L2_edges)))
    else:
        freq_L2 = 0
        freq_sum_L2 = 0

    # calculate relative frequency difference for edge
    rel_freq_dif = divide(freq_L1,freq_sum_L1) - divide(freq_L2,freq_sum_L2)

    #print('L1: {} to {}, freq {}, sum {}'.format(edge[0], edge[1], freq_L1, freq_sum_L1))
    #print('L2: {} to {}, freq {}, sum {}'.format(edge[0], edge[1], freq_L2, freq_sum_L2))

    # color of edge
    color = color_edge(rel_freq_dif)

    # calculate width
    if (not edge in L1_edges) and (not edge in L2_edges):
        width = 0.25
    else:
        width = divide(freq_L1,freq_sum_L1)*2 + divide(freq_L2,freq_sum_L2)*2 + 0.5
    
    #print('L1: {} to {}, width {}'.format(edge[0], edge[1], width))

    return rel_freq_dif, width, freq_L1, freq_L2, color


def get_node_label(activity, log1_act_count, log1_rel_act_count, log2_act_count, log2_rel_act_count, name_a, name_b):
    """
    Determines node label and color

    Parameters
    -----------
    activity: str
        Name of activity
    log1_act_count: dict
        Activities and absolute frequencies in log 1
    log1_rel_act_count: dict
        Activities and relative frequencies in log 1
    log2_act_count: dict
        Activities and absolute frequencies in log 2
    log2_rel_act_count: dict
        Activities and relative frequencies in log 2
    name_a: str
        Name of input log 1 
    name_b: str
        Name of input log 2
    
    Returns
    -----------
        (node_label, node_color): tuple
            Label of the node for visualization and color of the node for visualization
    """

    if activity in log1_act_count.keys():
        rel_freq_l1 = "{}%".format(log1_rel_act_count[activity])
        log1_act_num =  log1_act_count[activity]
        rel_freq_num1 = log1_rel_act_count[activity]
    else:
        rel_freq_l1 = "0%"
        log1_act_num = 0
        rel_freq_num1 = 0
    if activity in log2_act_count.keys():
        rel_freq_l2 = "{}%".format(log2_rel_act_count[activity])
        log2_act_num = log2_act_count[activity]
        rel_freq_num2 = log2_rel_act_count[activity]
    else:
        rel_freq_l2 = "0%"
        log2_act_num = 0
        rel_freq_num2 = 0
    rel_diff_vert, node_color = get_rel_freq_diff_vert(rel_freq_num1, rel_freq_num2)

    node_label = "\n{}:{} ({}), {}:{} ({})\n{}%".format(name_a, log1_act_num, rel_freq_l1, name_b, log2_act_num, rel_freq_l2, rel_diff_vert)

    return (node_label, node_color)

def get_edge_label(edge, dfg1, dfg2, num_cases_1, num_cases_2, log1, log2, name_a, name_b):
    """
    Determines edge label, color and width

    Parameters
    -----------
    edge: tuple
        (preceeding activity, succeeding activity)
    dfg1: dict
        DFG of log 1
    dfg2: dict
        DFG of log 2
    num_cases_1: int
        Number of cases in log 1
    num_cases_2: int
        Number of cases in log 2
    log1: DataFrame
        input log 1 
    log2: DataFrame
        input log 2
    name_a: str
        Name of input log 1 
    name_b: str
        Name of input log 2

    Returns
    -----------
    (edge_label, edge_color, width): triple
        Label of the edge, color of the edge, width of the edge for visualization
    """

    rel_freq_dif, width, freq_L1, freq_L2, edge_color = relative_freq_diff_width(edge, dfg1, dfg2)
    rel_freq_dif_perc = round(rel_freq_dif * 100,1)
    case_cov_1 = get_case_coverage_edge(log1, edge[0], edge[1], num_cases_1)
    case_cov_2 = get_case_coverage_edge(log2, edge[0], edge[1], num_cases_2)
    edge_label = f"{name_a}:{freq_L1} ({case_cov_1}%) {name_b}:{freq_L2} ({case_cov_2})% \n{rel_freq_dif_perc}%"
    return (edge_label, edge_color, width)


def get_start_end_edge_label(act, start_end_activities1, start_end_activities2, num_cases_1, num_cases_2, name_a, name_b):
    """
    Determines edge label of start/ end activity edges
    
    Parameters
    -----------
    act: str
        Name of activity
    start_end_activities1: dict
        Start/ end activities and frequencies in log 1
    start_end_activities2: dict
        Start/ end activities and frequencies in log 2
    num_cases_1: int
        Number of cases in log 1
    num_cases_2: int
        Number of cases in log 2
    name_a: str
        Name of input log 1 
    name_b: str
        Name of input log 2

    Returns
    ----------- 
    (edge_label): tuple
        Label of the edge for visualization
    """
    if act in start_end_activities1.keys():
        freq_L1 = start_end_activities1[act]
    else:
        freq_L1 = 0
    if act in start_end_activities2.keys():
        freq_L2 = start_end_activities2[act]
    else:
        freq_L2 = 0
    case_cov_1 = round(divide(freq_L1,num_cases_1)*100, 1)
    case_cov_2 = round(divide(freq_L2,num_cases_2)*100, 1)
    edge_label = f"{name_a}:{freq_L1} ({case_cov_1}%) \n{name_b}:{freq_L2} ({case_cov_2})%"

    return (edge_label)

def adapted_vpc_pipeline(log, log1, log2, threshold, name_a, name_b):
    """
    Evaluates logs based on adapted VPC procedure

    Parameters
    -----------
    log: event log/ DataFrame
        Concatenated event logs
    log1: event log/ DataFrame
        First input log
    log2: event log/ DataFrame
        Second input log
    threshold: float
        Minimum relative frequency for paths to be shown
    name_a: str
        Name of first log that will be displayed in visualization
    name_b:str 
        Name of second log that will be displayed in visualization

    Returns
    -----------
    dfg, act_f, start_f, end_f: dicts
        Filtered DFG of concatenated logs
    node_labels, node_colors, node_thickness: dicts
        Node properties to be displayed in visualizaton
    edge_labels, edge_colors, edge_thickness: dicts 
        Edge properties to be displayed in visualization
    start_labels, end_labels: dicts
        Edge labels for edges from start and to end
    """
    # convert to df
    log1_df = pm4py.convert_to_dataframe(log1)
    log2_df = pm4py.convert_to_dataframe(log2)
    # discover dfgs
    print("Discover and filter DFG...")
    dfg_full, start_activities, end_activities = pm4py.discover_dfg(log)
    print("Full DFG has {} edges".format(len(dfg_full)))
    activities = log['concept:name'].value_counts(dropna=False).keys().tolist()
    counts = log['concept:name'].value_counts(dropna=False).tolist()
    act_count = dict(zip(activities, counts))
    # filter dfg on top x% of paths, keep all activities
    dfg, start_f, end_f, act_f = pm4py.algo.filtering.dfg.dfg_filtering.filter_dfg_on_paths_percentage(dfg_full, start_activities, end_activities, act_count, threshold, keep_all_activities=True)
    print("Filtered DFG has {} edges".format(len(dfg)))
    # discover dfg of sublog 1
    dfg1, start_activities1, end_activities1 = pm4py.discover_dfg(log1)
    # get relative activity occurence
    log1_act = log1_df['concept:name'].unique()
    # activity count log 1
    log1_act_count = log_attributes.get_attribute_values(log1, 'concept:name', parameters={'keep_once_per_case':True})
    log1_rel_act_count = {key_val:round(log1_act_count[key_val] / len(log1_df['case:concept:name'].unique())*100,1) for key_val in log1_act_count.keys()}
    print("Discovered DFG 1")
    dfg2, start_activities2, end_activities2 = pm4py.discover_dfg(log2)
    # get relative activity occurence
    log2_act = log2_df['concept:name'].unique()
    log2_act_count = log_attributes.get_attribute_values(log2, "concept:name", parameters={'keep_once_per_case':True})
    log2_rel_act_count = {key_val:round(log2_act_count[key_val] / len(log2_df['case:concept:name'].unique())*100,1) for key_val in log2_act_count.keys()}
    print("Discovered DFG 2")
    # num cases of sublogs + num cases per log
    sum_cases = len(log1_df['case:concept:name'].unique()) + len(log2_df['case:concept:name'].unique())
    num_cases_1=len(log1_df['case:concept:name'].unique())
    print("Number of cases log 1: {}".format(num_cases_1))
    num_cases_2=len(log2_df['case:concept:name'].unique())
    print("Number of cases log 2: {}".format(num_cases_2))
    
    print("Creating annotations...")
    # create annotations
    node_labels = {act:get_node_label(act, log1_act_count, log1_rel_act_count, log2_act_count, log2_rel_act_count, name_a, name_b)[0] for act in activities}
    node_colors = {act:get_node_label(act, log1_act_count, log1_rel_act_count, log2_act_count, log2_rel_act_count, name_a, name_b)[1] for act in activities}
    node_thickness = {act:node_thickness(log1_act_count[act] if act in log1_act else 0, log2_act_count[act] if act in log2_act else 0, sum_cases) for act in activities}
    edge_labels = {}
    edge_colors = {}
    edge_thickness = {}
    num_all_edge = len(dfg.keys())
    i=1
    log1_conv = log_converter.apply(log1, variant=log_converter.Variants.TO_EVENT_LOG)
    log2_conv = log_converter.apply(log2, variant=log_converter.Variants.TO_EVENT_LOG)
    for edge in dfg.keys():
        edge_res = get_edge_label(edge, dfg1, dfg2, num_cases_1, num_cases_2, log1_conv, log2_conv, name_a, name_b)
        edge_labels[edge] = edge_res[0]
        edge_colors[edge] = edge_res[1]
        edge_thickness[edge] = edge_res[2]
        #print("{}/{}".format(i, num_all_edge))
        i+=1


    start_labels={act:get_start_end_edge_label(act, start_activities1, start_activities2,num_cases_1, num_cases_2, name_a, name_b) for act in start_activities}
    end_labels={act:get_start_end_edge_label(act, end_activities1, end_activities2,num_cases_1, num_cases_2, name_a, name_b) for act in end_activities}


    return dfg, act_f, start_f, end_f, node_labels, node_colors, node_thickness, edge_labels, edge_colors, edge_thickness, start_labels, end_labels


def pyvis_visualization(nt, activities_count, dfg, start_activities=None, end_activities=None, node_labels=None, node_colors=None, edge_labels=None, edge_colors=None, edge_thickness=None, node_thickness=None, 
                        show_edge_labels=True, edge_label_min=0, start_labels=None, end_labels=None, case_coverage_min=0):
    """
    Creates pyvis visualization of shared DFG 

    Parameters
    -----------
    nt: Network object 
        Empty Network
    activities_count: dict
        Activities in DFG along with their count
    dfg: dict
        DFG of concatenated logs
    start_activities: dict
        Start activities of the log
    end_activities: dict
        End activities of the log
    node_labels: dict
        Labels with relative frequency of activity in sub-logs
    edge_labels: dict
        Labels with relative frequency of edges and relative frequency difference of sub-logs
    node_colors: dict
        Colors of the nodes
    edge_colors: dict
        Colors of the edges
    edge_thickness: dict
        Thickness of edges
    node_thickness: dict
        Thickness of nodes
    show_edge_labels: boolean
        Whether to show or hide edge labels
    edge_label_min: float
        Minimum frequency difference of edge to be displayed as label
    start_labels: dict
        Labels with case coverage of start activities to edges
    end_labels: dict
        Labels with case coverage of edges to end activities
    case_coverage: float
        Minimum case coverage of edge to be displayed as label

    Returns
    -----------
    nt: Network object
        Network filled with nodes and edges
    """

    if start_activities is None:
        start_activities = []
    if end_activities is None:
        end_activities = []

    activities_count_int = copy(activities_count)

    activities_in_dfg = set(activities_count)


    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        activities_to_include = sorted(list(set(activities_in_dfg)))

    activities_map = {}

    # add start node
    nt.add_node("Start", "Start", shape='circle', color="lightgrey", physics=False)

    # add all nodes except for start and end nodes
    for i, act in enumerate(activities_to_include):
        label = node_labels[act]
        freq_diff = float(label.split("\n")[-1].strip("%").rstrip())
        if (abs(freq_diff) > 45) or (freq_diff < -25):
            node_font_col = '#ffffff'
            nt.add_node(str(hash(act)), label= act + node_labels[act], shape='box', color=node_colors[act], physics=False, borderWidth=node_thickness[act], font={'color':node_font_col})
        else:
            nt.add_node(str(hash(act)), label= act + node_labels[act], shape='box', color=node_colors[act], physics=False, borderWidth=node_thickness[act])
        activities_map[act] = str(hash(act))
    
    # add end node
    nt.add_node("End", "End", shape='circle', color="lightgrey", physics=False)

    # add edges in same order
    dfg_edges = sorted(list(dfg.keys()))

    # connect start to starting activities
    for act in start_activities:
        freq_e1 = float(start_labels[act].split(" ")[1].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())
        freq_e2 = float(start_labels[act].split(" ")[3].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())

        if start_labels is None or ((freq_e1 < case_coverage_min) and (freq_e2 < case_coverage_min)): 
            nt.add_edge("Start", str(hash(act)), dashes=True)
        else:
            nt.add_edge("Start", str(hash(act)),label=start_labels[act], dashes=True)

    # add edges
    for edge in dfg_edges:
        label = edge_labels[edge]
        freq_diff = float(label.split(" ")[-1].strip("%").rstrip())
        freq_e1 = float(label.split(" ")[1].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())
        freq_e2 = float(label.split(" ")[3].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())
        if (show_edge_labels == True) and (abs(freq_diff) >= edge_label_min) or ((freq_e1 >= case_coverage_min) or (freq_e2 >= case_coverage_min)):
            nt.add_edge(str(hash(edge[0])), str(hash(edge[1])), label=label, color=edge_colors[edge], width=edge_thickness[edge])
        else:
            nt.add_edge(str(hash(edge[0])), str(hash(edge[1])), color=edge_colors[edge], width=edge_thickness[edge])

    # connect end activities to end
    for act in end_activities:
        freq_e1 = float(end_labels[act].split(" ")[1].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())
        freq_e2 = float(end_labels[act].split(" ")[3].strip("%").replace("%","").replace("(","").replace(")","").rstrip().lstrip())
        if end_labels is None or ((freq_e1 < case_coverage_min) and (freq_e2 < case_coverage_min)):
            nt.add_edge(str(hash(act)), "End", dashes=True)
        else:
            nt.add_edge(str(hash(act)), "End", label=end_labels[act], dashes=True)

    print("Adapted VPC done!")
    return nt

def apply_adapted_vpc(nt, concat_log, log_1, log_2, name_a="A", name_b="B", frac_paths=0.4, show_edge_labels=True, edge_label_min=0, case_coverage_min=0):
    """
    Applies adapted VPC pipeline and visualization procedure

    Parameters
    -----------
    nt: Network object 
        Empty Network
    concat_log: event log/ DataFrame
        Concatenated event logs
    log_1: event log/ DataFrame
        Input log 1
    log_2: event log/ DataFrame
        Input log 2
    name_a: str
        Name for input log 1 for visualization
    name_b: str
        Name for input log 2 for visualization
    frac_paths: float
        Filter that determines which percentage of most common paths in DFG is shown
    show_edge_labels: boolean
        Whether to show or hide edge labels
    edge_label_min: float
        Minimum frequency difference of edge to be displayed as label
    case_coverage: float
        Minimum case coverage of edge to be displayed as label

    Returns
    -----------
    nt: Network object
        Network filled with nodes and edges
    """

    dfg, act, start, end, node_labels, node_colors, node_thickness, edge_labels, edge_colors, edge_thickness, start_labels, end_labels = adapted_vpc_pipeline(concat_log, log_1, log_2, frac_paths, name_a, name_b)

    nt = pyvis_visualization(nt, act, dfg,start_activities=start, end_activities=end, node_labels=node_labels, node_colors=node_colors,  edge_labels=edge_labels, edge_colors=edge_colors, 
                             node_thickness=node_thickness, edge_thickness=edge_thickness, show_edge_labels=show_edge_labels, edge_label_min=edge_label_min, start_labels=start_labels, end_labels=end_labels, case_coverage_min=case_coverage_min)
    
    return nt