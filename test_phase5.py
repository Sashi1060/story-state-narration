import copy
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.database import ROOT
from src.title_structure import (load_vocabulary, debrand, extract_structure, key_for, extract_features,
    set_similarity, sequence_similarity, structural_pairs, template_summary, channel_matrix, validation_sample,
    load_existing_pairs)


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.vocab = load_vocabulary(ROOT / 'config/narrative_vocabulary.json')
        self.titles = ['My wife cheated on me with my brother, so I chose her sister.',
                       'My fiancée left me for my best friend, so I married her cousin.',
                       'Reborn: I returned and they regret it!', 'A pleasant afternoon']
        self.frame = pd.DataFrame(dict(video_id=['a', 'b', 'c', 'd'], channel_id=['x', 'y', 'x', 'y'],
            channel_name=['X', 'Y', 'X', 'Y'], video_title=self.titles,
            video_views=[100, 200, 0, None], video_likes=[1, 2, 0, None], video_comments=[0, 1, None, None]))
        self.lookup = {(a,b):dict(semantic_similarity=.5, lexical_tfidf_similarity=.1, lexical_jaccard_similarity=.1)
                       for i,a in enumerate('abcd') for b in 'abcd'[i+1:]}

    def test_conservative_branding(self):
        self.assertEqual(debrand('Story - Jinwoo Recap', 'Jinwoo Recap')[:2], ('Story', True))
        for title in ['Jinwoo Recap visits town', 'Story - Jinwoo Reca[', 'Story about Jinwoo Recap', 'Story - jinwoo recap']:
            self.assertFalse(debrand(title, 'Jinwoo Recap')[1])
        self.assertEqual(debrand('茶 | Café', 'Café')[0], '茶')

    def test_obvious_events_and_negation(self):
        extraction = extract_structure('Reborn: I returned and they regret it!', self.vocab)
        self.assertEqual(json.loads(extraction['event_sequence_json']), ['REBIRTH', 'RETURN', 'REGRET'])
        for title in ['I did not cheat', 'I didn’t cheat', 'A pleasant afternoon', '', '花園で散歩']:
            self.assertEqual(json.loads(extract_structure(title, self.vocab)['event_sequence_json']), [])
        self.assertIsNone(extract_structure('I have a system', self.vocab)['structural_template'])

    def test_requested_examples_and_generalization(self):
        expected = 'ROMANTIC_BETRAYAL > CLOSE_ASSOCIATE_INVOLVEMENT > REPLACEMENT_RELATIONSHIP'
        for title in self.titles[:2] + ['My husband left me for my cousin, so I chose his brother.']:
            extraction = extract_structure(title, self.vocab)
            self.assertEqual(extraction['structural_template'], expected)
            self.assertTrue(json.loads(extraction['evidence_json']))

    def test_set_and_order_similarity(self):
        self.assertEqual(set_similarity(['A','B'], ['B','C']), 1/3)
        self.assertIsNone(set_similarity([], []))
        self.assertEqual(sequence_similarity(['A','B'], ['A','B']), 1)
        self.assertEqual(sequence_similarity(['A','B'], ['B','A']), .5)
        self.assertIsNone(sequence_similarity([], ['A']))

    def test_overlapping_labels_do_not_create_false_sequence(self):
        result=extract_structure('They discovered I was a billionaire CEO',self.vocab)
        self.assertEqual(json.loads(result['event_sequence_json']),['WEALTH_REVEAL'])
        self.assertIsNone(result['structural_template'])
        result=extract_structure('My family betrayed me',self.vocab)
        self.assertEqual(json.loads(result['event_sequence_json']),['BETRAYAL'])

    def test_cache_invalidation_and_reuse(self):
        original = key_for('a','Title','Title',self.vocab)
        self.assertNotEqual(original,key_for('b','Title','Title',self.vocab))
        self.assertNotEqual(original,key_for('a','New title','Title',self.vocab))
        self.assertNotEqual(original,key_for('a','Title','New title',self.vocab))
        changed = copy.deepcopy(self.vocab)
        changed['version'] = '2'
        self.assertNotEqual(original,key_for('a','Title','Title',changed))
        with tempfile.TemporaryDirectory() as folder:
            first, hits = extract_features(self.frame,self.vocab,folder)
            second, hits2 = extract_features(self.frame,self.vocab,folder)
            self.assertEqual(hits,0)
            self.assertEqual(hits2,4)
            pd.testing.assert_frame_equal(first,second)

    def test_pairs_exact_roles_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            features,_ = extract_features(self.frame,self.vocab,folder)
        pairs = structural_pairs(features,self.lookup)
        self.assertEqual(len(pairs),6)
        self.assertFalse(pairs.video_id_a.eq(pairs.video_id_b).any())
        match = pairs.iloc[0]
        self.assertTrue(match.exact_template_match)
        self.assertEqual(match.structural_sequence_similarity,1)
        self.assertEqual(match.structural_role_similarity,1)
        self.assertTrue(pairs[pairs.video_id_b.eq('d')].exact_template_match.isna().all())
        summary = template_summary(features)
        self.assertEqual(summary.iloc[0].video_count,2)
        matrix = channel_matrix(features,pairs)
        self.assertEqual(matrix.compared_video_pairs.sum(),6)
        self.assertEqual(matrix.exact_template_match_count.sum(),1)

    def test_single_empty_and_unknown_templates(self):
        with tempfile.TemporaryDirectory() as folder:
            for n in (0,1):
                features,_ = extract_features(self.frame.iloc[:n],self.vocab,folder)
                pairs = structural_pairs(features,{})
                self.assertTrue(pairs.empty)
                self.assertIn('structural_event_jaccard',pairs)
                self.assertTrue(validation_sample(pairs).empty)
        self.assertIsNone(extract_structure('Reborn',self.vocab)['structural_template'])

    def test_validation_sample_blank_and_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            features,_=extract_features(self.frame,self.vocab,folder)
        pairs=structural_pairs(features,self.lookup)
        sample=validation_sample(pairs)
        self.assertEqual(len(sample),6)
        self.assertTrue(sample.human_same_structure.eq('').all())
        self.assertTrue(sample.human_notes.eq('').all())
        pd.testing.assert_frame_equal(sample,validation_sample(pairs))

    def test_stale_similarity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            features,_=extract_features(self.frame,self.vocab,folder)
            pairs=structural_pairs(features,self.lookup)
            path=Path(folder)/'pairs.csv'
            pairs.to_csv(path,index=False,encoding='utf-8-sig')
            self.assertEqual(len(load_existing_pairs(path,self.frame)),6)
            pairs.loc[0,'title_a']='Changed title'
            pairs.to_csv(path,index=False,encoding='utf-8-sig')
            with self.assertRaises(ValueError):
                load_existing_pairs(path,self.frame)


if __name__ == '__main__':
    unittest.main()
