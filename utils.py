####################### MODULES #######################
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from config import CODE, AA_2_NUM, POS_ALIGN_Yip, SEQ_3TGI_YIP,SEQ_CHYMO_YIP,HEDSTROM_SWAP,IND_3TGI_PYMOL,SEQ_3TGI

#######################################################

#######################################################

def plot_hist_energies_with_seq(model_path,align=None,Nmax_seq=5000):
	output = np.load(model_path,allow_pickle=True)[()]
	J,h = Zero_Sum_Gauge(output['J'],output['h'])

	if align is None:
		align = output['align']
	if align.shape[0]> Nmax_seq:
		ind_rand = np.random.choice(np.arange(align.shape[0]),Nmax_seq,replace=False)
		sub_align =align[ind_rand]
	else:
		sub_align=align
	Energies = compute_energies(sub_align,h,J)

	plt.figure(figsize=(10,5))

	range_E = (-590,-300)
	plt.hist(Energies, bins=60, range=range_E, histtype='step', color='black', linewidth=3.2)

	seq_tryp = np.array([AA_2_NUM[SEQ_3TGI_YIP[i]] for i in range(len(SEQ_3TGI_YIP))])
	seq_chymo = np.array([AA_2_NUM[SEQ_CHYMO_YIP[i]] for i in range(len(SEQ_CHYMO_YIP))])
	seq_D189S = np.copy(seq_tryp)
	seq_S189D = np.copy(seq_chymo)

	ind_189 = np.where(POS_ALIGN_Yip=='189')[0][0]
	seq_D189S[ind_189] = AA_2_NUM[SEQ_CHYMO_YIP[ind_189]]
	seq_S189D[ind_189] = AA_2_NUM[SEQ_3TGI_YIP[ind_189]]
	seq_Hedstrom = np.copy(seq_tryp)
	for idx in HEDSTROM_SWAP:
		ind_sho = np.where(POS_ALIGN_Yip==idx)[0][0]
		seq_Hedstrom[ind_sho] = AA_2_NUM[SEQ_CHYMO_YIP[ind_sho]]

	E_Hedstrom = compute_energies(np.expand_dims(seq_Hedstrom,axis=0),h,J)[0]
	E_tryp = compute_energies(np.expand_dims(seq_tryp,axis=0),h,J)[0]
	E_chymo = compute_energies(np.expand_dims(seq_chymo,axis=0),h,J)[0]
	E_189 = compute_energies(np.expand_dims(seq_D189S,axis=0),h,J)[0]

	lw = 2.8
	plt.vlines(x=E_tryp,ymin=0,ymax=np.max(plt.gca().get_ylim()),linestyle='--',color = 'black',linewidth=lw,label='Rat Trypsin')
	plt.vlines(x=E_189,ymin=0,ymax=np.max(plt.gca().get_ylim()),linestyle='--',color = 'blue',linewidth=lw,label='Rat Trypsin D189S')
	plt.vlines(x=E_Hedstrom,ymin=0,ymax=np.max(plt.gca().get_ylim()),linestyle='--',color = 'red',linewidth=lw,label='Hedstrom\'s swap')
	plt.vlines(x=E_chymo,ymin=0,ymax=np.max(plt.gca().get_ylim()),linestyle='--',color = 'grey',linewidth=lw,label='Bovin Chymo.')

	plt.xlabel(r'Statistical energies', fontsize=18)
	plt.ylabel(r'Occurences', fontsize=18)
	plt.yticks([])
	plt.legend(loc='upper right',fontsize=18,frameon=False)
	plt.xlim(range_E)
	plt.xticks(fontsize=18)

	for spine in ["top", "right"]:
		plt.gca().spines[spine].set_visible(False)
	plt.gca().spines['bottom'].set_linewidth(1.7)
	plt.gca().spines['left'].set_linewidth(1.7)

	plt.show()


#######################################################

#######################################################

def Delta_VS_DMS(path_mod,path_dms,Mutations=None,SEQ_REF = SEQ_3TGI,CompMut=None):
	Exp,Pred = plot_matrix_DMS_prediction(path_mod,path_dms)

	fig = plt.figure(figsize=(8, 8))

	scatter_ax = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
	hist_x_ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=scatter_ax)
	hist_y_ax = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=scatter_ax)

	scatter_ax.scatter(Pred, Exp, color='black', alpha=0.7, s=50)

	scatter_ax.axvline(x=0, color='red', linestyle='--', linewidth=1.8)
	scatter_ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1.8)

	scatter_ax.set_xlabel(r'$\Delta E$', fontsize=18)
	scatter_ax.set_ylabel('Log enrichment', fontsize=18)

	scatter_ax.tick_params(axis='both', labelsize=18)
	scatter_ax.grid(True, linestyle="dashed", linewidth=0.5, alpha=0.5)

	for spine in ["top", "right"]:
		scatter_ax.spines[spine].set_visible(False)

	hist_x_ax.hist(Pred.flatten(), bins=50, color='black', alpha=0.7)
	hist_y_ax.hist(Exp.flatten(), bins=50, orientation='horizontal', color='black', alpha=0.7)

	hist_x_ax.axis("off")
	hist_y_ax.axis("off")

	if CompMut is not None:
		Ind_223 = [np.where(IND_3TGI_PYMOL==m[1])[0][0] for m in CompMut if m[1] in IND_3TGI_PYMOL and m[0]!='-']
		Mutations_plot = [AA_2_NUM[m[0]]-1 for m in CompMut if m[1] in IND_3TGI_PYMOL and m[0]!='-']

		scatter_ax.scatter(Pred[Mutations_plot,Ind_223], Exp[Mutations_plot,Ind_223], color='blue', s=50,label=r'Compensatory mut.')

	if Mutations is not None:
		seq_ref = np.array([AA_2_NUM[SEQ_REF[i]] for i in range(len(SEQ_REF))])
		for Mut in Mutations:
			Ind_mut = np.where(IND_3TGI_PYMOL==Mut[1])[0][0]
			Mut_aa = AA_2_NUM[Mut[0]]-1
			scatter_ax.scatter(Pred[Mut_aa,Ind_mut], Exp[Mut_aa,Ind_mut], color='red', s=50,label=str(CODE[seq_ref[Ind_mut]])+str(Mut[1])+Mut[0])
		
	if CompMut is not None or Mutations is not None:
		scatter_ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True)

	plt.show()


def plot_matrix_DMS_prediction(path_mod,path_dms):
	Mut_DMS = np.load(path_dms)

	#### PLOT PREDICTION ####
	output = np.load(path_mod,allow_pickle=True)[()]
	J,h = Zero_Sum_Gauge(output['J'],output['h'])

	seq_Ecoli = np.array([AA_2_NUM[SEQ_3TGI_YIP[i]] for i in range(len(SEQ_3TGI_YIP))])

	Mut_pred = Mutational_effect(seq_Ecoli,h,J)
	Mut_pred[seq_Ecoli,np.arange(Mut_pred.shape[1])] = np.nan

	Mut_plot = Mut_pred[1:,(seq_Ecoli!=0)]

	return Mut_DMS,Mut_plot

#######################################################

########### Compensatory mutations with DE ############

def Propose_Mutation_DE(path_mod,Nb_mut,Mutations,Np=10,MSA_name='Yip',SEQ_REF = None):
	output = np.load(path_mod,allow_pickle=True)[()]
	J,h = output['J'],output['h']
	J,h = Zero_Sum_Gauge(J,h)

	if MSA_name == 'Yip':
		POS_ALIGN = POS_ALIGN_Yip
		if SEQ_REF is None:
			SEQ_REF = SEQ_3TGI_YIP
	else: 
		POS_ALIGN = np.arange(1,h.shape[0]+1)
		if SEQ_REF is None:
			raise ValueError("A reference sequence (SEQ_REF) must be provided for prediction.")
	
	if len(SEQ_REF) != h.shape[0]:
		raise ValueError("The alignment length must match the length of the reference sequence.")

	seq_tryp = np.array([AA_2_NUM[SEQ_REF[i]] for i in range(len(SEQ_REF))])

	Ind_proposed = []
	for Mut in Mutations:
		seq_mut = np.copy(seq_tryp)
		seq_mut[np.where(POS_ALIGN==Mut[1])[0][0]] = AA_2_NUM[Mut[0]]

		Ind_proposed.append((AA_2_NUM[Mut[0]],np.where(POS_ALIGN==Mut[1])[0][0]))

	Mut_pred_WT = Mutational_effect(seq_tryp,h,J)

	seq_ref = np.copy(seq_mut)

	if Nb_mut>0:
		for mut in range(Nb_mut):

			Mut_pred = Mutational_effect(seq_ref,h,J)

			for ind in Ind_proposed:
				Mut_pred[:,ind[1]] = np.nan

			ind_min = np.unravel_index(np.nanargmin(Mut_pred, axis=None), Mut_pred.shape)
			Ind_proposed.append(ind_min)
			sig_mut = ind_min[0]

			if mut < Np:
				print('Mutation '+str(mut + 2)+': '+str(CODE[seq_ref[ind_min[1]]])+str(POS_ALIGN[ind_min[1]])+str(CODE[sig_mut])+'   ΔE = '+str(Mut_pred[ind_min]))
			seq_ref[ind_min[1]] = sig_mut

	return Ind_proposed

def Convert_CompMut_in_3TGI_Indices(Ind_mut,SEQ_REF = SEQ_3TGI_YIP):
	Mutations = []
	for ind in Ind_mut:
		Mutations.append((CODE[ind[0]],POS_ALIGN_Yip[ind[1]]))
	return Mutations

####################### TOOLS #########################

def compute_energies(seqs,h,J=None):
	"""
	Function to compute energies for an alignment based on the provided parameters provided h and J values.

	Args:
	- align (numpy.array): A 2D or 1D numpy array representing the input alignment.
	- h (numpy.array): A 2D numpy array representing the h values (fields).
	- J (numpy.array): A 4D numpy array representing the J values (couplings).

	Returns:
	- numpy.array: A 1D numpy array representing the computed energies for the input alignment.
	"""
	if len(seqs.shape)==2:
		L=seqs.shape[1]
	elif len(seqs.shape)==1:
		L=seqs.shape[0]
		seqs=seqs.reshape((1,L))
	if J is None:
		J = np.zeros((L,L,h.shape[1],h.shape[1]))
	energy=np.sum(np.array([h[i,seqs[:,i]] for i in range(L)]),axis=0)
	energy=energy+(np.sum(np.array([[J[i,j,seqs[:,i],seqs[:,j]] for j in range(L)] for i in range(L)]),axis=(0,1))/2)
	return -energy

def Zero_Sum_Gauge(J,h):
	"""
	Function to apply a zero-sum gauge transformation to J and h matrices.
	"""
	if J is None:
		L,q = h.shape
		J = np.zeros((L,L,q,q))
		h_zg = np.copy(h)
		h_zg -= np.expand_dims(np.mean(h,axis = 1),axis = 1) 
		h_zg += np.sum(np.mean(J,axis=3)-np.expand_dims(np.mean(J,axis=(2,3)),axis=2),axis=1)
		return None,h_zg
	
	if h is None:
		L,q = J.shape[0],J.shape[2]
		J_zg = np.copy(J)
		J_zg -= np.expand_dims(np.mean(J,axis = 2),axis = 2) 
		J_zg -= np.expand_dims(np.mean(J,axis=3),axis =3) 
		J_zg += np.expand_dims(np.mean(J,axis=(2,3)),axis=(2,3))
		return J_zg, None

	J_zg = np.copy(J)
	h_zg = np.copy(h)

	h_zg -= np.expand_dims(np.mean(h,axis = 1),axis = 1) 
	h_zg += np.sum(np.mean(J,axis=3)-np.expand_dims(np.mean(J,axis=(2,3)),axis=2),axis=1)

	J_zg -= np.expand_dims(np.mean(J,axis = 2),axis = 2) 
	J_zg -= np.expand_dims(np.mean(J,axis=3),axis =3) 
	J_zg += np.expand_dims(np.mean(J,axis=(2,3)),axis=(2,3))

	return J_zg, h_zg

def Mutational_effect(WT_seq,h,J):
	"""
	Function to compute mutational effects based on the provided wild-type sequence, h, and J values, 
	considering the changes in energy resulting from single amino acid substitutions.
	"""
	L,q = h.shape
	Mut = np.zeros((q,L))
	for k in range(L):
		for b in range(q):
			if b != WT_seq[k]:
				Delta_E = h[k,WT_seq[k]] - h[k,b] - np.sum(J[k,np.arange(L),b,WT_seq]) + np.sum(J[k,np.arange(L),WT_seq[k],WT_seq]) 
				Mut[b,k] = Delta_E
	return Mut

def Jw(W,q):
	L=int(((q*q-2*q)+((2*q-q*q)**2+8*W.shape[0]*q*q)**(1/2))/2/q/q)
	J=np.zeros((L,L,q,q))
	h=np.zeros((L,q))
	x=np.array([[i,j] for i,j in it.combinations(range(L),2)])
	for a in range(q):
		for b in range(q):
			J[x[:,0],x[:,1],a,b]=W[(q**2*((x[:,0])*(2*L-x[:,0]-1)/2+(x[:,1]-x[:,0]-1))+(a)*q+b).astype(int)]
			J[x[:,1],x[:,0],b,a]=J[x[:,0],x[:,1],a,b]
	x=np.array(range(L))
	for a in range(q):
		h[x[:],a]=W[(q**2*L*(L-1)/2+q*x[:]+a).astype(int)]
	return J,h
