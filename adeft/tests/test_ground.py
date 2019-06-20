import os
import json
import flask
import shutil
import logging
import tempfile
import unittest

from adeft.gui.ground import create_app
from adeft.locations import TEST_RESOURCES_PATH

logger = logging.getLogger(__name__)


class GroundingTestCase1(unittest.TestCase):
    """Test grounding gui when no initial groundings are provided."""
    def setUp(self):
        longforms = ['ionizing radiation', 'insulin receptor',
                     'insulin resistance', 'irradiation']
        scores = [20.0, 15.0, 10.0, 7.0]
        grounding_map = {longform: '' for longform in longforms}
        names_map = {longform: '' for longform in longforms}
        pos_labels = []
        outpath = os.path.join(TEST_RESOURCES_PATH, 'scratch')
        verbose = False
        port = 5000
        app = create_app(longforms, scores, grounding_map, names_map,
                         pos_labels, outpath, verbose, port)
        app.testing = True
        self.longforms = longforms
        self.grounding_map = grounding_map
        self.names_map = names_map
        self.outpath = outpath
        self.app = app

    # Tests
    def test_init(self):
        """Test that page is loaded correctly."""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res

            assert flask.session['names_map'] == self.names_map
            assert flask.session['grounding_map'] == self.grounding_map

    def test_add_groundings1(self):
        """Test adding a single name, grounding pair"""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res

            names_map = self.names_map.copy()
            grounding_map = self.grounding_map.copy()

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '1'})
            assert res.status_code == 200, res
            names_map['insulin receptor'] = 'INSR'
            grounding_map['insulin receptor'] = 'HGNC:6091'

            assert names_map == flask.session['names_map']
            assert grounding_map == flask.session['grounding_map']

    def test_add_groundings2(self):
        """Test adding same name and grounding to multiple longforms."""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res

            names_map = self.names_map.copy()
            grounding_map = self.grounding_map.copy()

            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res

            names_map['ionizing radiation'] = 'Radiation, Ionizing'
            names_map['irradiation'] = 'Radiation, Ionizing'
            grounding_map['ionizing radiation'] = 'MESH:D011839'
            grounding_map['irradiation'] = 'MESH:D011839'

            assert names_map == flask.session['names_map']
            assert grounding_map == flask.session['grounding_map']

    def test_add_groundings3(self):
        """Test entering successive groundings."""
        with self.app.test_client() as tc:
            tc.get('/')
            names_map = self.names_map.copy()
            grounding_map = self.grounding_map.copy()
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6090',
                                'select': '1'})
            assert res.status_code == 200, res

            names_map['ionizing radiation'] = 'Radiation, Ionizing'
            names_map['irradiation'] = 'Radiation, Ionizing'
            names_map['insulin receptor'] = 'INSR'
            grounding_map['ionizing radiation'] = 'MESH:D011839'
            grounding_map['irradiation'] = 'MESH:D011839'
            grounding_map['insulin receptor'] = 'HGNC:6090'

            assert names_map == flask.session['names_map']
            assert grounding_map == flask.session['grounding_map']

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '1'})
            assert res.status_code == 200, res

            grounding_map['insulin receptor'] = 'HGNC:6091'
            assert grounding_map == flask.session['grounding_map']

    def test_add_and_delete_groundings(self):
        """Test deletion of groundings"""
        with self.app.test_client() as tc:
            tc.get('/')
            names_map = self.names_map.copy()
            grounding_map = self.grounding_map.copy()
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6090',
                                'select': '1'})
            assert res.status_code == 200, res

            res = tc.post('ground_delete',
                          data={'delete.1': 'X'})
            assert res.status_code == 200, res

            names_map['ionizing radiation'] = 'Radiation, Ionizing'
            names_map['irradiation'] = 'Radiation, Ionizing'
            grounding_map['ionizing radiation'] = 'MESH:D011839'
            grounding_map['irradiation'] = 'MESH:D011839'

            assert names_map == flask.session['names_map']
            assert grounding_map == flask.session['grounding_map']

            # test that deletion does nothing when there are no groundings
            res = tc.post('ground_delete',
                          data={'delete.2': 'X'})
            assert names_map == flask.session['names_map']
            assert grounding_map == flask.session['grounding_map']

    def test_set_pos_labels(self):
        """Test setting of positive labels"""
        with self.app.test_client() as tc:
            tc.get('/')
            # Add groundings for ionizing radiation, insulin receptor
            # and insulin resistance
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '1'})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007333',
                                'select': '2'})

            # Labels are the unique groundings stored in sorted order
            # pos labels contains a list of strings representing ints
            res = tc.post('ground_pos_label',
                          data={'pos-label.0': '+'})
            assert res.status_code == 200, res

            assert flask.session['pos_labels'] == [0]

            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1]

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1, 2]

            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 2]

    def test_add_delete_pos_labels_interaction(self):
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '1'})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007333',
                                'select': '2'})

            assert res.status_code == 200, res
            # Labels are the unique groundings stored in sorted order
            # pos labels contains a list of strings representing ints
            res = tc.post('ground_pos_label',
                          data={'pos-label.0': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1, 2]

            res = tc.post('ground_delete',
                          data={'delete.2': 'X'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1]

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007332',
                                'select': '2'})
            assert flask.session['pos_labels'] == [0, 2]

            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert flask.session['pos_labels'] == [0, 1, 2]

            tc.post('ground_add',
                    data={'name': 'Insulin Resistance',
                          'grounding': 'MESH:D007333',
                          'select': '2'})
            assert flask.session['pos_labels'] == [0, 2]

    def test_generate_grounding_map(self):
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['0', '3']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '1'})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007333',
                                'select': '2'})

            assert res.status_code == 200, res
            # Labels are the unique groundings stored in sorted order
            # pos labels contains a list of strings representing ints
            res = tc.post('ground_pos_label',
                          data={'pos-label.0': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res

            tc.post('generate_grounding_map')

            # Get output from temporary file
            outpath = self.outpath
            with open(os.path.join(outpath, 'output.json')) as f:
                output = json.load(f)

            os.remove(os.path.join(outpath, 'output.json'))

            grounding_map = {'ionizing radiation': 'MESH:D011839',
                             'irradiation': 'MESH:D011839',
                             'insulin receptor': 'HGNC:6091',
                             'insulin resistance': 'MESH:D007333'}
            assert output['grounding_map'] == grounding_map

            names = {'MESH:D011839': 'Radiation, Ionizing',
                     'MESH:D007333': 'Insulin Resistance',
                     'HGNC:6091': 'INSR'}
            assert output['names'] == names

            pos_labels = ['HGNC:6091', 'MESH:D011839']
            assert pos_labels == output['pos_labels']
