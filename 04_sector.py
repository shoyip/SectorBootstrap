import numpy as np
from termcolor import colored
import time
import sys

from cocoatree.io import load_MSA
from cocoatree.statistics.pairwise import compute_sca_matrix
from cocoatree.deconvolution import extract_independent_components, extract_xcors

def get_sequence_sectors(sequence, sectors, colors=['red', 'green', 'blue']):
    colored_string_list = []
    for residue_index, residue in enumerate(sequence):
        boolean_list = [residue_index in sector for sector in sectors]
        if np.sum(boolean_list) == 1:
            for boolean, color in zip(boolean_list, colors):
                if boolean:
                    colored_string_list.append(colored(residue, color))
        elif np.sum(boolean_list) == 0:
            colored_string_list.append(residue)
        else:
            print('error')
    return print(''.join(colored_string_list))

if __name__ == "__main__":
    dataset = list(map(np.array, load_MSA('./data/iter_aln_dedup_sp.faa').values()))
    descriptions, sequences = dataset
    weights = np.load('./data/full_weights.npy', allow_pickle=True).ravel()
    M = len(descriptions)
    rat_trypsin = "IVGGYTCQENSVPYQVSLNS-----GYHFCGGSLINDQWVVSAAHCYKS-------RIQVRLGEHNIN-VLEGNEQFVNAAKIIKHPNFDR--KTLNNDIMLIKLSSPVKLNARVATV-ALPS---SCAP-AG-TQCLISGWGNTLSSG----VNEPDLLQCLDAPLLPQADCEASYP--GKITDNMVCVGFLEGGKDSCQGDSGGPVVCN-----GELQGIVSWGY--GCALPDNPGVYTKVCNYVDWIQDTIAAN---"
    nums = ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '37', '37a', '37b', '37c', '37d', '37e', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '61a', '61b', '61c', '61d', '61e', '61f', '61g', '62', '63', '64', '66', '67', '68', '69', '70', '71', '72', '73', '74', '74a', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '96a', '96b', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '121a', '122', '123', '124', '125', '125a', '125b', '125c', '127', '128', '129', '130', '130a', '132', '133', '133a', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '148a', '148b', '148c', '148d', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '173a', '173b', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '184A', '185', '186', '187', '188', '188A', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '202a', '202b', '202c', '202d', '202e', '203', '204', '209', '210', '211', '212', '213', '214', '215', '216', '217', '217a', '217b', '219', '220', '221', '221A', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '245a', '245b', '245c']
    
    time_start = time.time()
    sca_matrix = compute_sca_matrix(sequences, weights)
    time_end = time.time()
    
    print("It took", time_end - time_start, "seconds and", sys.getsizeof(sca_matrix) / 1024**2, "MB.")
    
    sectors = extract_xcors(sca_matrix)

    get_sequence_sectors(rat_trypsin, sectors, colors=['green', 'red', 'blue'])

    print("The RED SECTOR indices in order of relevance is:", ', '.join(map(str, sectors[1])))
    red_sector_indices = (sectors[1])
    print("In the TRYPSIN NUMBERING we have", ', '.join(np.array(nums)[red_sector_indices]))
