import flask
import tempfile
import unittest

from adeft.gui.ground import create_app
from adeft.gui import _empty_grounding_map


class GroundingTestCase1(unittest.TestCase):
    """Test grounding gui when no initial groundings are provided."""
    def setUp(self):
        longforms = ['ionizing radiation', 'insulin receptor',
                     'insulin resistance', 'irradiation']
        scores = [20.0, 15.0, 10.0, 7.0]
        grounding_map = _empty_grounding_map(longforms)
        names_map = {}
        pos_labels = []
        outpath = tempfile.mkdtemp()
        verbose = False
        port = 5000
        app = create_app(longforms, scores, grounding_map, names_map,
                         pos_labels, outpath, verbose, port)
        app.testing = True
        self.app = app

    # Tests
    def test_init(self):
        """Test that page is loaded correctly."""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            assert flask.session['names'] == ['', '', '', '']
            assert flask.app.session['groundings'] == ['', '', '', '']

    def test_add_groundings1(self):
        """Test adding a single name, grounding pair"""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '2'})
            assert res.status_code == 200, res
            assert flask.session['names'] == ['', 'INSR', '', '']
            assert flask.session['groundings'] == ['', 'HGNC:6091', '', '']

    def test_add_groundings2(self):
        """Test adding same name and grounding to multiple longforms."""
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['1', '4']})
            assert res.status_code == 200, res
            assert flask.session['names'] == ['Radiation, Ionizing', '', '',
                                              'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', '', '',
                                                   'MESH:D011839']

    def test_add_groundings3(self):
        """Test entering successive groundings."""
        with self.app.test_client() as tc:
            tc.get('/')
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['1', '4']})
            assert res.status_code == 200, res

            # Add incorrect grounding for insulin receptor
            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6090',
                                'select': '2'})
            assert res.status_code == 200, res
            assert flask.session['names'] == ['Radiation, Ionizing', 'INSR',
                                              '', 'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', 'HGNC:6090',
                                                   '', 'MESH:D011839']
            # resubmit with correct grounding
            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '2'})
            assert res.status_code == 200, res
            assert flask.session['names'] == ['Radiation, Ionizing', 'INSR',
                                              '', 'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', 'HGNC:6091',
                                                   '', 'MESH:D011839']

    def test_add_and_delete_groundings(self):
        """Test deletion of groundings"""
        with self.app.test_client() as tc:
            tc.get('/')
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['1', '4']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6090',
                                'select': '2'})
            assert res.status_code == 200, res
            assert flask.session['names'] == ['Radiation, Ionizing', 'INSR',
                                              '', 'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', 'HGNC:6090',
                                                   '', 'MESH:D011839']

            # Delete incorrect grounding for insulin receptor
            res = tc.post('ground_delete',
                          data={'delete.2': 'X'})
            assert res.status_code == 200, res

            assert flask.session['names'] == ['Radiation, Ionizing', '', '',
                                              'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', '', '',
                                                   'MESH:D011839']

            # test that deletion does nothing when there are no groundings
            res = tc.post('ground_delete',
                          data={'delete.3': 'X'})
            assert flask.session['names'] == ['Radiation, Ionizing', '', '',
                                              'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', '', '',
                                                   'MESH:D011839']

    def test_set_pos_labels(self):
        """Test setting of positive labels"""
        with self.app.test_client() as tc:
            tc.get('/')
            # Add groundings for ionizing radiation, insulin receptor
            # and insulin resistance
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['1', '4']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '2'})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007333',
                                'select': '3'})
                          
            assert res.status_code == 200, res
            assert flask.session['names'] == ['Radiation, Ionizing', 'INSR',
                                              'Insulin Resistance',
                                              'Radiation, Ionizing']
            assert flask.session['groundings'] == ['MESH:D011839', 'HGNC:6091',
                                                   'MESH:D007333',
                                                   'MESH:D011839']

            # Labels are the unique groundings stored in sorted order
            # pos labels contains a list of strings representing ints
            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0]

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1]

            res = tc.post('ground_pos_label',
                          data={'pos-label.3': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1, 2]

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 2]

    def test_add_delete_pos_labels_interaction(self):
        with self.app.test_client() as tc:
            res = tc.get('/')
            assert res.status_code == 200, res
            res = tc.post('ground_add',
                          data={'name': 'Radiation, Ionizing',
                                'grounding': 'MESH:D011839',
                                'select': ['1', '4']})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'INSR',
                                'grounding': 'HGNC:6091',
                                'select': '2'})
            assert res.status_code == 200, res

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007333',
                                'select': '3'})

            assert res.status_code == 200, res
            # Labels are the unique groundings stored in sorted order
            # pos labels contains a list of strings representing ints
            res = tc.post('ground_pos_label',
                          data={'pos-label.1': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_pos_label',
                          data={'pos-label.3': '+'})
            assert res.status_code == 200, res

            res = tc.post('ground_delete',
                          data={'delete.3': 'X'})
            assert res.status_code == 200, res
            assert flask.session['pos_labels'] == [0, 1]

            res = tc.post('ground_add',
                          data={'name': 'Insulin Resistance',
                                'grounding': 'MESH:D007332',
                                'select': '3'})
            assert flask.session['pos_labels'] == [0, 2]

            res = tc.post('ground_pos_label',
                          data={'pos-label.2': '+'})
            assert flask.session['pos_labels'] == [0, 1, 2]

            tc.post('ground_add',
                    data={'name': 'Insulin Resistance',
                          'grounding': 'MESH:D007333',
                          'select': '3'})
            print(flask.session['pos_labels'])
            assert flask.session['pos_labels'] == [0, 1, 2]

            tc.post('ground_add',
                    data={'name': 'Insulin Resistance',
                          'grounding': 'MESH:D999999',
                          'select': '3'})
            print(flask.session['pos_labels'])
            assert flask.session['pos_labels'] == [0, 1, 2]

            
