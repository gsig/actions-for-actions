# Contributors:
# Jonghyun Choi
# Dustin Schwenk
# Gunnar Sigurdsson
from __future__ import division
import numpy as np
import pandas as pd
from base_evaluator import BaseEvaluator


class CharadesEvaluator(BaseEvaluator):
    n_classes = 157

    def __init__(self, data_path, submission_path, subset=None):
        super(CharadesEvaluator, self).__init__(data_path, submission_path)
        self.submission_columns = ['frame_id']
        self.gt_labels = None
        self.vid_ids = None
        self.gt_array = None
        self.submission = None
        self.submission_array = None
        self.subtask_name = None
        self.normalize_map = True # This version normlizes the number of positives and negatives for each class
        self.N_all = None 
        self.F_all = None 
        self.subset = set(subset) if subset is not None else None
        self.all_subtasks = ['temporal_segmentation', 'action_recognition']

    def load_groundtruth(self):
        gt_labels = pd.read_csv(self.data_path)
        if self.subset is not None:
            mask = [True if x in self.subset else False for x in gt_labels['id'].values]
            gt_labels = gt_labels[mask]
            assert np.any(np.array(mask))
        gt_labels['length'] = pd.to_numeric(gt_labels['length'])
        gt_labels['actions'].fillna('', inplace=True)
        self.gt_labels = gt_labels

    def load_submission(self, submission_file):
        loc_submission = pd.read_csv(submission_file, header=None)
        build_proc_sub = loc_submission[0].str.split(' ').values.tolist()
        assert len(build_proc_sub[0]) == self.n_classes + len(self.submission_columns)
        proc_sub = pd.DataFrame.from_records(build_proc_sub, columns=[self.submission_columns + list(range(self.n_classes))])
        if self.subset is not None:
            mask = [True if x in self.subset else False for x in proc_sub['frame_id'].values]
            proc_sub = proc_sub[mask]
            assert np.any(np.array(mask))
        num_proc_sub = proc_sub.apply(pd.to_numeric, errors='ignore')
        grouped_by_vid = num_proc_sub
        self.submission = grouped_by_vid

    def check_frame_count(self):
        assert self.submission.shape[0] == self.gt_array.shape[0]

    def check_complete(self, submission):
        #self.check_frame_count()
        gt_vids = set(self.vid_ids)
        #submitted_vids = set(submission['frame_id'].tolist())
        submitted_vids = set(submission['frame_id'].values)
        vids_missing = gt_vids.difference(submitted_vids)
        vids_extra = submitted_vids.difference(gt_vids)
        #assert not vids_missing and not vids_extra
        if vids_missing or vids_extra:
            print('Warning: Number of scores in submission file does not match ground truth. Attempting to continue')

    def compute_scores(self):
        m_aps = []
        for oc_i in range(self.n_classes):
            sorted_idxs = np.argsort(- self.submission_array[:, oc_i])
            tp = self.gt_array[:, oc_i][sorted_idxs] == 1
            fp = np.invert(tp)
            n_pos = tp.sum()
            if n_pos < 0.1:
                m_aps.append(float('nan'))
                continue
            n_neg = fp.sum()
            f_pcs = np.cumsum(fp)
            t_pcs = np.cumsum(tp)
            prec = t_pcs / (f_pcs + t_pcs)
            if self.normalize_map:
                k = self.N_all/n_pos
                k2 = self.F_all/n_neg
                prec=(t_pcs*k) / (f_pcs*k2+t_pcs*k)
            avg_prec = 0
            for i in range(self.submission_array.shape[0]):
                if tp[i]:
                    avg_prec += prec[i]
            m_aps.append(avg_prec / n_pos)
        m_aps = np.array(m_aps)
        m_ap = np.mean(m_aps)
        w_ap = (m_aps * self.gt_array.sum(axis=0) / self.gt_array.sum()).sum()
        return m_ap, w_ap, m_aps

    def build_ground_truth_array(self):
        pass

    def build_aligned_submission_array(self):
        pass

    def evaluate_submission(self):
        self.load_groundtruth()
        self.load_submission(self.submission_path)
        self.build_ground_truth_array()
        self.validate_submission(self.submission)
        self.build_aligned_submission_array()

        mean_ap, weighted_ap, m_aps = self.compute_scores()
        return self.subtask_name, mean_ap, m_aps


class LocalizationEvaluator(CharadesEvaluator):
    n_classes = CharadesEvaluator.n_classes
    n_frames = 25

    def __init__(self, data_path, submission_path, subset=None):
        super(LocalizationEvaluator, self).__init__(data_path, submission_path, subset)
        self.subtask_name = 'temporal_segmentation'
        self.submission_columns.append('frame_n')
        self.N_all = 1100
        self.F_all = 45000

    @staticmethod
    def check_time_range(timepoints, start, stop):
        return start <= timepoints <= stop

    def build_gt_vectors(self, act_seq, vid_len, time_checker):
        frame_actions = np.zeros((self.n_frames, self.n_classes))
        time_seq = np.linspace(0, vid_len, self.n_frames, endpoint=False)
        act_seq = act_seq.split(';')
        if not act_seq[0]:
            return frame_actions
        for act in act_seq:
            act_id, start, end = act.split(' ')
            start = float(start)
            end = float(end)
            act_idx = int(act_id[1:])
            activated_time_idxs = time_checker(time_seq, start, end)
            frame_actions[activated_time_idxs, [act_idx]] = 1
        return frame_actions

    def build_ground_truth_array(self):
        check_times_vectorized = np.vectorize(self.check_time_range)
        img_arrays = []
        vid_ids = []
        for i in range(self.gt_labels.shape[0]):
            row = self.gt_labels.iloc[i]
            vid_ids.append(row['id'])
            img_gt_arr = self.build_gt_vectors(row['actions'], row['length'], check_times_vectorized)
            img_arrays.append(img_gt_arr)
        comb_gt_array = np.vstack(img_arrays)
        self.vid_ids = vid_ids
        self.gt_array = comb_gt_array

    def build_aligned_submission_array(self):
        aligned_submission_array = np.ones((len(self.vid_ids) * self.n_frames, self.n_classes))
        #self.submission = self.submission.groupby('frame_id')
        self.submission.groupby(self.submission['frame_id'].values)
        for g_idx, g in enumerate(self.vid_ids):
            #submission_img_df = self.submission.get_group(g)
            #sort_submission_img_arr = submission_img_df.sort_values('frame_n').values[:, 2:]
            submission_img_df = self.submission[(self.submission['frame_id'] == g).values]
            sort_submission_img_arr = submission_img_df.sortlevel(['frame_id','frame_n']).values[:, 2:]
            aligned_submission_array[g_idx * self.n_frames: (g_idx + 1) * self.n_frames:] = sort_submission_img_arr
        self.submission_array = aligned_submission_array


class ClassificationEvaluator(CharadesEvaluator):
    n_classes = CharadesEvaluator.n_classes
    n_frames = 1

    def __init__(self, data_path, submission_path, subset=None):
        super(ClassificationEvaluator, self).__init__(data_path, submission_path, subset)
        self.subtask_name = 'action_recognition'
        self.N_all = 104 
        self.F_all = 1759 

    def build_gt_vectors(self, act_seq):
        frame_actions = np.zeros(self.n_classes)
        act_seq = act_seq.split(';')
        if not act_seq[0]:
            return frame_actions
        for act in act_seq:
            act_id, _, _ = act.split(' ')
            act_idx = int(act_id[1:])
            frame_actions[act_idx] = 1
        return frame_actions

    def build_ground_truth_array(self):
        img_arrays = []
        vid_ids = []
        for i in range(self.gt_labels.shape[0]):
            row = self.gt_labels.iloc[i]
            vid_ids.append(row['id'])
            img_gt_arr = self.build_gt_vectors(row['actions'])
            img_arrays.append(img_gt_arr)
        comb_gt_array = np.vstack(img_arrays)
        self.vid_ids = vid_ids
        self.gt_array = comb_gt_array

    def build_aligned_submission_array(self):
        aligned_submission_array = np.ones((len(self.vid_ids) * self.n_frames, self.n_classes))
        for g_idx, g in enumerate(self.vid_ids):
            #submission_img_arr = self.submission[self.submission['frame_id'] == g].values[:, 1:]
            submission_img_arr = self.submission[(self.submission['frame_id'] == g).values].values[:, 1:]
            if not self.gt_labels[self.gt_labels['id'] == g]['actions'].item():
                submission_img_arr = - np.inf * np.ones_like(submission_img_arr)
            if submission_img_arr.shape[0]==0:
                submission_img_arr = - np.inf * np.ones((1,self.n_classes))
            aligned_submission_array[g_idx * self.n_frames: (g_idx + 1) * self.n_frames:] = submission_img_arr
        self.submission_array = aligned_submission_array

