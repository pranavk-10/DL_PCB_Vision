"""Exercise schema, mapping, bounds, ambiguous paths, and source-group leakage guards."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from config.config import CONFIG
from src.dataset_loader import Annotation, DatasetError, PCBRecord, assert_no_leakage, load_dataset, parse_annotations, split_groups, validate_box

class DatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pcb_loader_test_')
        self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def annotation_file(self,text):
        path=self.root/'annotations.txt';path.write_text(text,encoding='utf-8');return path
    def record(self,index,group=None):
        return PCBRecord(str(index),group or f'group{index}',self.root/f'{index}_test.jpg',self.root/f'{index}_temp.jpg',self.root/f'{index}.txt',(Annotation((0,0,3,3),0,1),))
    def test_all_six_labels_and_nonmatching_display_order(self):
        annotations=parse_annotations(self.annotation_file('\n'.join(f'0 0 10 10 {label}' for label in range(1,7))))
        self.assertEqual([value.class_id for value in annotations],[0,1,2,3,5,4])
    def test_bom_and_blank_lines(self):
        self.assertEqual(len(parse_annotations(self.annotation_file('\ufeff\n1 2 5 8 1\n\n'))),1)
    def test_empty_normal_annotation(self):
        self.assertEqual(parse_annotations(self.annotation_file('')),())
    def test_malformed_row_rejected(self):
        with self.assertRaises(DatasetError):parse_annotations(self.annotation_file('1 2 3 4'))
    def test_noninteger_rejected(self):
        with self.assertRaises(DatasetError):parse_annotations(self.annotation_file('0 0 2.5 3 1'))
    def test_unknown_label_rejected(self):
        with self.assertRaises(DatasetError):parse_annotations(self.annotation_file('0 0 5 5 7'))
    def test_reversed_and_negative_boxes_rejected(self):
        for row in ('-1 0 3 3 1','5 0 2 3 1','0 4 3 4 1'):
            with self.subTest(row=row),self.assertRaises(DatasetError):parse_annotations(self.annotation_file(row))
    def test_one_based_inclusive_conversion(self):
        self.assertEqual(parse_annotations(self.annotation_file('1 1 10 12 6'),'one')[0].box,(0,0,10,12))
    def test_unknown_origin_rejected(self):
        with self.assertRaises(DatasetError):parse_annotations(self.annotation_file('0 0 3 3 1'),'guess')
    def test_box_boundary_validation(self):
        validate_box((0,0,640,640),640,640)
        for box in ((-1,0,3,3),(0,0,641,640),(0,0,3,641),(2,2,2,3)):
            with self.subTest(box=box),self.assertRaises(DatasetError):validate_box(box,640,640)
    def test_source_group_split_is_reproducible(self):
        records=[self.record(index//2*2+index%2,f'group{index//2}') for index in range(20)]
        first=split_groups(records);second=split_groups(list(reversed(records)))
        for name in first:self.assertEqual({r.group_id for r in first[name]},{r.group_id for r in second[name]})
        assert_no_leakage(first)
        self.assertTrue(all(first.values()))
        self.assertEqual(sum(map(len,first.values())),len(records))
    def test_pair_and_group_leakage_detected(self):
        record=self.record(1)
        with self.assertRaises(DatasetError):assert_no_leakage({'train':[record],'test':[record]})
        with self.assertRaises(DatasetError):assert_no_leakage({'train':[record],'test':[self.record(2,record.group_id)]})
    def test_shared_template_path_leakage_detected(self):
        first=self.record(1);second=replace(self.record(2),template_path=first.template_path)
        with self.assertRaises(DatasetError):assert_no_leakage({'train':[first],'test':[second]})
    def test_too_few_groups_rejected(self):
        with self.assertRaises(DatasetError):split_groups([self.record(1),self.record(2)])
    def test_actual_directory_discovery_with_synthetic_paths(self):
        group=self.root/'group00001';images=group/'00001';annotations=group/'00001_not'
        images.mkdir(parents=True);annotations.mkdir()
        (images/'00001000_test.jpg').write_bytes(b'not decoded in discovery')
        (images/'00001000_temp.jpg').write_bytes(b'not decoded in discovery')
        (annotations/'00001000.txt').write_text('0 0 5 5 1')
        records=load_dataset(self.root,strict=True)
        self.assertEqual(len(records),1);self.assertEqual(records[0].group_id,'group00001')
        self.assertEqual(records[0].annotation_path,annotations/'00001000.txt')
    def test_missing_pair_rejected_gracefully(self):
        (self.root/'a_test.jpg').write_bytes(b'fixture')
        with self.assertRaises(DatasetError):load_dataset(self.root)
    def test_missing_dataset_helpful_error(self):
        with self.assertRaisesRegex(DatasetError,'does not exist'):load_dataset(self.root/'absent')
    def test_ambiguous_annotation_not_guessed(self):
        (self.root/'a_test.jpg').write_bytes(b'fixture');(self.root/'a_temp.jpg').write_bytes(b'fixture')
        for name in ('one','two'):
            folder=self.root/name;folder.mkdir();(folder/'a.txt').write_text('0 0 5 5 1')
        with self.assertRaises(DatasetError):load_dataset(self.root,strict=True)
if __name__=='__main__':unittest.main()
