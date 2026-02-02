####################### MODULES #######################
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
from config import CODE, AA_2_NUM, POS_ALIGN_Yip, SEQ_3TGI_YIP

#######################################################

####################### Statistics ####################

def Stats_Pos_bis(align, Pos,ref_pos=None, maxp=5,perc=True,MSA_name='Yip'):
	if MSA_name == 'Yip':
		POS_ALIGN = POS_ALIGN_Yip
	else: 
		POS_ALIGN = np.arange(1,align.shape[1]+1)
	
	list_aa = align[:, np.where(POS_ALIGN == Pos)[0]]

	if ref_pos is not None:
		list_aa_ref = align[:, np.where(POS_ALIGN == ref_pos[1])[0]]
		ind_ref = (list_aa_ref==AA_2_NUM[ref_pos[0]])
		list_aa_withref = list_aa[ind_ref]
		unique, counts = np.unique(list_aa_withref, return_counts=True)
	else:
		unique, counts = np.unique(list_aa, return_counts=True)

	sorted_data = sorted(zip(counts, unique), reverse=True)
	counts, unique = zip(*sorted_data)
	if len(counts) > maxp:
		top_counts = list(counts[:maxp])
		top_labels = [CODE[aa] for aa in unique[:maxp]]
		others_count = sum(counts[maxp:])
		top_counts.append(others_count)
		top_labels.append('Others')
	else:
		top_counts = list(counts)
		top_labels = [CODE[i] for i in unique]

	# Trier les données par ordre décroissant
	sorted_data = sorted(zip(top_counts, top_labels), reverse=True)
	top_counts, top_labels = zip(*sorted_data)
	top_counts,top_labels = list(top_counts),list(top_labels)

	percentages = [100 * top_counts[i] / sum(top_counts) for i in range(len(top_counts)) if top_labels[i]!='Others']
	top_counts = [top_counts[i] for i in range(len(top_counts)) if top_labels[i]!='Others']
	top_labels = [top_labels[i] for i in range(len(top_labels)) if top_labels[i]!='Others']

	plt.figure(figsize=(6, 4))
	bars = plt.barh(range(len(top_counts)), top_counts, color='black')

	plt.ylabel("Amino Acids", fontsize=18)
	plt.yticks(range(len(top_labels)), top_labels, fontsize=16)

	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['bottom'].set_visible(False)

	plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

	for i, (count, percentage) in enumerate(zip(top_counts, percentages)):
		if perc:
			plt.text(count, i, f'  {percentage:.1f}%', va='center',fontsize=16)
		else: plt.text(count, i, f'  {count}', va='center',fontsize=16)

	plt.gca().invert_yaxis() 
	plt.tight_layout()

	plt.show()

#######################################################


####################### COMBINE #######################


def Propose_Mutation_DDE2(path_mod,Nb_mut,Mutations,Np=10,POS_ALIGN=None,SEQ_REF=None):
	output = np.load(path_mod,allow_pickle=True)[()]
	J,h = output['J'], output['h']
	J,h = ut.Zero_Sum_Gauge(J,h)

#	if MSA_name == 'Yip':
#		POS_ALIGN = POS_ALIGN_Yip
#		if SEQ_REF is None:
#			SEQ_REF = SEQ_3TGI_YIP
#	else: 
#		POS_ALIGN = np.arange(1,h.shape[0]+1)
#		if SEQ_REF is None:
#			raise ValueError("A reference sequence (SEQ_REF) must be provided for prediction.")
#	
#	if len(SEQ_REF) != h.shape[0]:
#		raise ValueError("The alignment length must match the length of the reference sequence.")

	seq_tryp = np.array([AA_2_NUM[SEQ_REF[i]] for i in range(len(SEQ_REF))])

	Ind_proposed = []
	for Mut in Mutations:
		seq_mut = np.copy(seq_tryp)
		seq_mut[np.where(POS_ALIGN==Mut[1])[0][0]] = AA_2_NUM[Mut[0]]

		Ind_proposed.append((AA_2_NUM[Mut[0]],np.where(POS_ALIGN==Mut[1])[0][0]))

	Mut_pred_WT = ut.Mutational_effect(seq_tryp,h,J)

	seq_ref = np.copy(seq_mut)

	if Nb_mut>0:
		for mut in range(Nb_mut):

			Mut_pred = ut.Mutational_effect(seq_ref,h,J)

			Mut_diff = Mut_pred - Mut_pred_WT

			for ind in Ind_proposed:
				Mut_diff[:,ind[1]] = np.nan

			ind_min = np.unravel_index(np.nanargmin(Mut_diff, axis=None), Mut_diff.shape)
			Ind_proposed.append(ind_min)
			sig_mut = ind_min[0]

			if mut < Np:
				print('Mutation '+str(mut + 2)+': '+str(CODE[seq_ref[ind_min[1]]])+str(POS_ALIGN[ind_min[1]])+str(CODE[sig_mut])+'   ΔΔE = '+str(Mut_diff[ind_min]))
			seq_ref[ind_min[1]] = sig_mut

	return Ind_proposed

#######################################################

