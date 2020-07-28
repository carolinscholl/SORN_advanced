import sys
sys.path.append('../../')

from NetworkBehaviour.Logic.SORN.SORN_advanced import *
#from NetworkBehaviour.Logic.SORN.SORN_advanced_buffer import *
from NetworkBehaviour.Input.Music.DrumBeatActivator import *
from NetworkCore.Network import *
from NetworkCore.Synapse_Group import *
from Testing.Common.Music_Helper import *
from NetworkBehaviour.Structure.Structure import *
from Exploration.StorageManager.StorageManager import *
#from Testing.Common.SORN_visualization import *

display = False
so = True

def run(tag, ind=[], par={'N_e':[1800], 'TS':[1]}):
    name = 'DrumBeats_'+'_'+tag+str(par['N_e'])+'N_e_'+str(par['TS'])

    sm = StorageManager(main_folder_name=tag, folder_name=name, random_nr=False, print_msg=display)
    sm.save_param_dict(par)

    source = DrumBeatActivator(tag='drum_act', which_tracks=[100], filter_silence=False, THR_similar_tracks=0.5, input_density=0.015, offtoken=True, ontoken=True, include_inverse_alphabet=False)#, include_inverse_alphabet= True)#output_size=par['N_e']

    inh_thr = ['0.1;+-0%','0.1;+-0%', '0.1;+-0%']
    exc_thr = ['0.1;+-100%', '0.1;+-100%', '0.1;+-100%']

    SORN = Network()

    for i, lag in enumerate(par['TS']):#
        e_ng = NeuronGroup(net=SORN, tag='exc_cell_{},prediction_source,text_input_group'.format(lag), size=get_squared_dim(int(par['N_e'][i])), behaviour={
            2: SORN_init_neuron_vars(iteration_lag=lag, init_TH=exc_thr[i]),
            3: SORN_init_afferent_synapses(transmitter='GLU', density='13%', distribution='lognormal(0,[0.89#0])', normalize=True, partition_compensation=True),
            4: SORN_init_afferent_synapses(transmitter='GABA', density='50%', distribution='lognormal(0,[0.80222#1])', normalize=True),

            12: SORN_slow_syn(transmitter='GLU', strength='[0.1383#2]', so=so),
            13: SORN_slow_syn(transmitter='GABA', strength='-[0.1698#3]', so=so),

            17: SORN_fast_syn(transmitter='GABA', strength='-[0.11045#4]', so=so),
            18: SORN_input_collect(),

            19: SORN_Refractory(factor='0.5;+-50%'),

            21: SORN_STDP(eta_stdp='[0.0001#5]', prune_stdp=False),
            22: SORN_SN(syn_type='GLU', clip_max=None, init_norm_factor=1.0),

            23: SORN_IP_TI(h_ip='lognormal_real_mean([0.04#6], [0.2944#7])', eta_ip='[0.0006#8];+-50%', integration_length='[30#18];+-50%', clip_min=None),#, gap_percent=10 #30;+-50% #0.0003
            #24: SORN_NOX(mp='np.mean(n.output_new)', eta_nox='[0.3#9];+-50%', behaviour_enabled=False), #0.4
            25: SORN_NOX(mp='self.partition_sum(n)', eta_nox='[0.3#9];+-50%'),  # 0.4

            26: SORN_SC_TI(h_sc='lognormal_real_mean([0.01#10], [0.2944#11])', eta_sc='[0.1#12];+-50%', integration_length='1'), #60;+-50% #0.05
            27: SORN_iSTDP(h_ip='same(SCTI, th)', eta_istdp='[0.0001#13]'),

            30: SORN_finish(),

            #99: SynapseRecorder(['[np.sum(s.slow_add)]'], tag='ex_glu_syn_rec'),
            #100: NeuronRecorder(['n.output'], tag='exc_out_rec')
        })

        i_ng = NeuronGroup(net=SORN, tag='inh_cell_{}'.format(lag), size=get_squared_dim(int(0.2 * (par['N_e'][i]))), behaviour={
            2: SORN_init_neuron_vars(iteration_lag=lag, init_TH=inh_thr[i]),
            3: SORN_init_afferent_synapses(transmitter='GLU', density='45%', distribution='lognormal(0,[0.87038#14])', normalize=True),  # 450
            4: SORN_init_afferent_synapses(transmitter='GABA', density='20%', distribution='lognormal(0,[0.82099#15])', normalize=True),  # 40

            #11: SORN_slow_syn(transmitter='GABA', strength='-[0.1838#16]', so=so),
            14: SORN_fast_syn(transmitter='GLU', strength='[1.5#16]', so=so),#1.5353
            15: SORN_fast_syn(transmitter='GABA', strength='-[0.08#17]', so=so),#0.08
            18: SORN_input_collect(),

            19: SORN_Refractory(factor='0.1;0.4'),

            #23: SORN_IP_TI(h_ip='lognormal_real_mean([0.08#6], [0.2944#7])', eta_ip='[0.0003#8];+-50%', integration_length='30;+-50%', clip_min=None),

            30: SORN_finish(),

            #100: NeuronRecorder(['n.output'], tag='inh_out_rec')
        })

        i_ng['structure', 0].stretch_to_equal_size(e_ng)


        SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU,ee', connectivity='(s_id!=d_id)*in_box(10)', partition=True)#.partition([10, 10], [4, 4])
        SynapseGroup(net=SORN, src=e_ng, dst=i_ng, tag='GLU,ie', connectivity='(s_id!=d_id)*in_box(10)', partition=True)#.partition([5, 5], [2, 2])
        SynapseGroup(net=SORN, src=i_ng, dst=e_ng, tag='GABA,ei', connectivity='(s_id!=d_id)*in_box(10)', partition=True)
        SynapseGroup(net=SORN, src=i_ng, dst=i_ng, tag='GABA,ii', connectivity='(s_id!=d_id)*in_box(10)', partition=True)

        if lag == 1:
            i_ng.add_behaviour(10, SORN_external_input(strength=1.0, pattern_groups=[source]))
            e_ng.add_behaviour(10, SORN_external_input(strength=1.0, pattern_groups=[source]))
            #e_ng.add_behaviour(101, NeuronRecorder(['n.pattern_index'], tag='inp_rec'))
        else:
            #forward synapses
            SynapseGroup(net=SORN, src=last_e_ng, dst=e_ng, tag='GLU,eeff')#.partition([10, 10], [4, 4])
            SynapseGroup(net=SORN, src=last_e_ng, dst=i_ng, tag='GABA,ieff')#.partition([5, 5], [2, 2])
            #backward synapses
            SynapseGroup(net=SORN, src=e_ng, dst=last_e_ng, tag='GLU,eebw')#.partition([10, 10], [4, 4])
            SynapseGroup(net=SORN, src=e_ng, dst=last_i_ng, tag='GABA,iebw')#.partition([5, 5], [2, 2])

        last_e_ng = e_ng # excitatory neuron group
        last_i_ng = i_ng # inhibitory neuron group
        e_ng.color = (0,0,255,255)
        i_ng.color = (255,0,255,255)
        e_ng.pattern_index=0
        i_ng.pattern_index=0

    SORN.set_marked_variables(ind, info=(ind == []))

    SORN.initialize(info=False)
    ############################################################################################################################################################

    score_spont = score_predict_train = score_predict_test = 0

    #import Exploration.UI.Network_UI.Network_UI as SUI
    #SUI.Network_UI(SORN, label='SORN UI default setup', storage_manager=sm, group_display_count=2, reduced_layout=False).show()

    SORN = run_plastic_phase(SORN, steps_plastic=1000000, display=True, storage_manager=sm)


    readout, X_train, Y_train, X_test, Y_test = train_readout(SORN, steps_train=100000, steps_test=100, source = SORN['drum_act', 0], display=True, stdp_off=True, storage_manager=sm)
    
    score_predict_train = get_score_predict_next_step(SORN, SORN['drum_act', 0],readout, X_train, Y_train, display=True, stdp_off=True, storage_manager=sm)

    score_predict_test = get_score_predict_next_step(SORN, SORN['drum_act', 0],readout, X_test, Y_test, display=True, stdp_off=True, storage_manager=sm)
    
    score_spont = get_score_spontaneous_music(SORN, SORN['drum_act', 0], readout, split_tracks=False, steps_spont=2000, display=True, stdp_off=True, 
    same_timestep_without_feedback_loop=False, steps_recovery=0, create_MIDI=True, storage_manager=sm)#, steps_recovery=15000
    
    sm.save_obj('score_train', score_predict_train)
    sm.save_obj('score_test', score_predict_test)
    sm.save_obj('score_spontaneous', score_spont)

    #plot_frequencies_poly(score_spont, path = sm.absolute_path+'frequencies', title='{} Ne, {} lag'.format(par['N_e'], par['TS']))

    return score_predict_train, score_predict_test, score_spont
    #return {'Prediction training set: ': score_predict_train, 'Prediction test set :': score_predict_test, 'Spontaneous:': score_spont}

if __name__ == '__main__':
    ind = []

    res_train, res_test, res_spont = run(tag='drum', ind=[0.95, 0.4, 0.1383, 0.1698, 0.1, 0.0015, 0.04, 0.2944, 0.0006, 0.2, 0.015, 0.2944, 0.1, 0.001, 0.87038, 0.82099, 1.5, 0.08, 15.0])
    print('score', res_spont['total_score'])

