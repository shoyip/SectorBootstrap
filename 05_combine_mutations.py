import numpy as np
import utils_COMBINE as uc
import matplotlib.pyplot as plt
from pathlib import Path
import utils as ut
import os

POS_ALIGN_Yip = np.array(['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '37', '37a', '37b', '37c', '37d', '37e', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '61a', '61b', '61c', '61d', '61e', '61f', '61g', '62', '63', '64', '66', '67', '68', '69', '70', '71', '72', '73', '74', '74a', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '96a', '96b', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '121a', '122', '123', '124', '125', '125a', '125b', '125c', '127', '128', '129', '130', '130a', '132', '133', '133a', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '148a', '148b', '148c', '148d', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '173a', '173b', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '184A', '185', '186', '187', '188', '188A', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '202a', '202b', '202c', '202d', '202e', '203', '204', '209', '210', '211', '212', '213', '214', '215', '216', '217', '217a', '217b', '219', '220', '221', '221A', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '245a', '245b', '245c'])

red_sector = [197, 239, 237, 224, 186, 225, 189, 190, 200, 227, 228, 222, 238, 2, 229, 164, 195, 194, 231, 165, 176, 1]
red_sector = sorted(red_sector)

marion_red_sector = [224,197,239,237,225,227,186,200,189,228,190,194,2,229,195,231,164,88,183,222,107,23,21]
marion_red_sector = sorted(marion_red_sector)

align_yip_red = POS_ALIGN_Yip[red_sector]
align_mar_red = POS_ALIGN_Yip[marion_red_sector]

SEQ_3TGI_YIP = 'IVGGYTCQENSVPYQVSLNS-----GYHFCGGSLINDQWVVSAAHCYKS-------RIQVRLGEHNIN-VLEGNEQFVNAAKIIKHPNFDR--KTLNNDIMLIKLSSPVKLNARVATV-ALPS---SCAP-AG-TQCLISGWGNTLSSG----VNEPDLLQCLDAPLLPQADCEASYP--GKITDNMVCVGFLEGGKDSCQGDSGGPVVCN-----GELQGIVSWGY--GCALPDNPGVYTKVCNYVDWIQDTIAAN---'
Seq = np.array([s for s in SEQ_3TGI_YIP])[marion_red_sector]
SEQ_3TGI_YIP_RedSec = "".join(list(Seq))

for i in range(10):
    print(f"===\nALIGNMENT N. {i}\n===")
    path_mod = f'./results/RedSecMar_SubAln_{i}/RedSecMar_SubAln_{i}_ModelSBM_N_chains50_N_iter400_Param_initzero_k_MCMC100000_lambda_J0.0_lambda_h0_m1_theta0.3_N_Av1_R0.npy'
    Nb_mut = 5
    CompMut_DDE = uc.Propose_Mutation_DDE2(
            path_mod,
            Nb_mut,
            [('S','189')],
            POS_ALIGN=align_mar_red,
            SEQ_REF=SEQ_3TGI_YIP_RedSec)
#CompMut_3TGI_numbering = ut.Convert_CompMut_in_3TGI_Indices(CompMut_DDE)
#print(CompMut_3TGI_numbering)
