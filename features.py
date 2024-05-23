#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 21:40
# @Author  : Bing
# @email    : 1747193328@qq.com
import collections
import json
import os.path
import math
from itertools import product, combinations
import pandas as pd
import pymatgen
import scipy
from joblib import Parallel, delayed
from matminer.featurizers.base import BaseFeaturizer
from monty.serialization import loadfn
from pymatgen.core import Composition, Element, Species, Structure, FloatWithUnit
from enum import Enum
import utils


class AtomsData:
    _instance = None
    init_flag = False

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, path="./dataset/inorganic_tables.json"):
        if AtomsData.init_flag:
            return
        AtomsData.init_flag = True
        with open(path, "r", encoding="utf8") as f:
            self.data = json.loads(f.read())

    def get_Valence(self, element_name, oxidation_state, orbit):
        if not isinstance(oxidation_state, str):
            oxidation_state = str(int(oxidation_state))
        valence = self.get_elem_property(element_name, "Valence")
        if oxidation_state in valence.keys():
            valence = valence[oxidation_state]
        else:
            return 0
        t = round(valence.get(orbit, 0))

        elem = Element(element_name)

        if orbit=="s":

            return t+  elem.row*0.1

        elif orbit=="p":

            return t+elem.row*0.1
        elif orbit=="d":
            return t+  (elem.row-1)*0.1
        return t + (elem.row-2)*0.1

    def mean(self, col: pd.Series, weight):
        if (col >= 0).all():
            return scipy.stats.gmean(col, weights=weight)

        elif (col < 0).all():

            return -scipy.stats.gmean(col.abs(), weights=weight)
        else:
            raise ValueError()

    def get_elements_propertys(self, species, weight=None, coordination=6):

        if isinstance(species, Species):
            species = [species]
        elif isinstance(species, str):
            species = [Species.from_str(species)]

        if weight is None:
            weight = [1 / len(species) for i in range(len(species))]
        else:
            if sum(weight) != 1:
                weight = [i / sum(weight) for i in weight]

        features_list = []
        for specie in species:
            features = []
            specie: Element
            features.append(specie.number)
            features.append(specie.group)
            features.append(specie.row)
            features.append(specie.atomic_mass)
            features.append(specie.ionization_energy)
            features.append(specie.molar_volume)
            features.append(self.get_elem_property(specie.element.name, "HeatOfFusion"))
            features.append(specie.mendeleev_no)
            features.append(specie.melting_point)
            features.append((self.get_shannon_radii(specie.element.name, specie.oxi_state, coordination)))
            features.append(self.get_elem_property(specie.element.name, "electric_pol"))
            features.append(specie.X)
            features.append(specie.electron_affinity)
            features.append(self.get_Valence(specie.element.name, specie.oxi_state, "s"))
            features.append(self.get_Valence(specie.element.name, specie.oxi_state, "p"))
            features.append(self.get_Valence(specie.element.name, specie.oxi_state, "d"))
            features.append(self.get_Valence(specie.element.name, specie.oxi_state, "f"))
            features.append(round(self.get_elem_property(specie.element.name, "Homo"), 2))
            features.append(round(self.get_elem_property(specie.element.name, "Lumo"), 2))
            features.append(round(self.get_elem_property(specie.element.name, "HomoLumoGap"), 2))
            features_list.append(features)
        features = pd.DataFrame(features_list, columns=self.lables).T
        mean = features.apply(lambda x: sum(x * weight), axis=1)
        features["means"] = mean
        return features.means

    @property
    def lables(self):
        lables = []
        lables.append(f"N")
        lables.append(f"C")
        lables.append(f"R")
        lables.append(f"M")
        lables.append(f"IE")
        lables.append(f"Mv")
        lables.append(f"Hf")
        lables.append(f"Mn")
        lables.append(f"Mp")
        lables.append(f"R^ion")
        lables.append(f"Ep")
        lables.append(f"X")
        lables.append(f"EA")
        lables.append(f"V^s")
        lables.append(f"V^p")
        lables.append(f"V^d")
        lables.append(f"V^f")
        lables.append(f"HOMO")
        lables.append(f"LUMO")
        lables.append(f"HLGap")
        return lables

    @utils.catch_ex
    def get_elem_property(self, elem_name: str, property: str):
        #
        result = self.data[elem_name][property]
        if isinstance(result, str):
            result = result.split(" ")
            if len(result) == 2:
                key, unit = result
                return FloatWithUnit(key, unit=unit)
        return result

    def get_shannon_radii(self, element_name: str, oxidation_state, coordination):
        if not isinstance(oxidation_state, str):
            oxidation_state = str(int(oxidation_state))
        if not isinstance(coordination, str):
            coordination = str(int(coordination))

        radiis = self.get_elem_property(element_name, "Shannon_radii")
        if oxidation_state in radiis.keys():
            Shannon_radii = radiis[oxidation_state]
        else:
            raise ValueError(
                f"element_name  {element_name} oxidation_state {oxidation_state} coordination {coordination}")

        if coordination in Shannon_radii.keys():
            return Shannon_radii[coordination]["only_spin"]
        else:
            for local_coordination in list(Shannon_radii.keys())[::-1]:
                if int(coordination) >= int(local_coordination):
                    return Shannon_radii[local_coordination]["only_spin"]
            return Shannon_radii[local_coordination]["only_spin"]


class PerovskiteEnum(Enum):
    Oxide = "氧化物钙钛矿"
    Halide = "卤化物钙钛矿"
    Sulfide = "硫化物钙钛矿"
    Organic = "有机-无机钙钛矿"


oxi_state = {
    "F": [-1],
    "Cl": [-1],
    "Br": [-1],
    "I": [-1],
    "O": [-2],
    "S": [-2],

}

with open("./dataset/inorganic_tables.json", "r", encoding="utf8") as f:
    data = json.loads(f.read())
    for elem, info in data.items():
        if elem not in oxi_state.keys():
            oxi_state[elem] = [int(i) for i in info.get("Shannon_radii", {}).keys() if int(i) < 4]


class Perovskite:
    ASiteNum = 1
    BSiteNum = 1
    XSiteNum = 3
    types = {
        16: [2, 4, -2],
        17: [1, 2, -1],

    }

    def __init__(self, system, all_oxi_states=False, oxi_states_override=None):
        if oxi_states_override:
            self.oxi_states = oxi_states_override
        else:
            self.oxi_states = oxi_state
        if isinstance(system, str):
            self.compostion = Composition(system).reduced_composition
        elif isinstance(system, Composition):
            self.compostion = system.reduced_composition
        elif isinstance(system, Structure):
            self.compostion = system.composition

        else:

            raise ValueError("The system must be str or Composition")

        self.type = 17
        self.atoms_source = AtomsData("./dataset/inorganic_tables.json")
        self.PARSE = False
        self.Organic = {}
        self.result = {
            "A": {"systems": [],
                  "fractions": [],
                  "valence": []
                  },
            "B": {"systems": [],
                  "fractions": [],
                  "valence": []},

            "X": {"systems": [],
                  "fractions": [],
                  "valence": []},
            "result": 0,
            "type": None
        }

        self.parse_sites(all_oxi_states)

    def set_result_value(self, site, elem, valence):
        self.result[site]["systems"].append(elem)
        frac = self.compostion[elem]
        self.result[site]["fractions"].append(frac)
        self.result[site]["valence"].append(valence)

    def set_sites(self, site_name, sites, valence):
        if isinstance(valence, list):
            i = 0
            for elem in sites:
                self.set_result_value(site_name, elem.name, valence[i])
                i += 1
        else:

            for elem in sites:
                self.set_result_value(site_name, elem.name, valence)

    def get_score(self, score_list=[]):
        score = []

        for i in score_list:
            score.append(sum(self.compostion[elem] for elem in i))

        return score

    def fill_elems(self, sets: list, target=1):

        result = set()
        frac_list = [self.compostion[elem] for elem in sets]
        if target in frac_list:

            result.add(sets[frac_list.index(target)])
        else:
            remain_sets = [elem for elem in sets if self.compostion[elem] <= target]

            for i in range(10):
                score = self.get_score([result])[0]

                diff = score - target

                if diff != 0:
                    result.add(remain_sets.pop(0))

                else:
                    break

        return result

    def parse_halide_perovskite(self, oxidation_states):
        # print(oxidation_states)
        a_set = self.elems.get(oxidation_states[0])
        if len(oxidation_states) != 1:
            b_set = self.elems.get(oxidation_states[-1])
        else:
            b_set = []

        elem_list_a = sorted(a_set,
                             key=lambda x: (self.atoms_source.get_shannon_radii(x.name, 1, 12)),
                             reverse=True)

        result = self.fill_elems(elem_list_a, self.ASiteNum)

        oxidation = [1 for i in result]
        self.set_sites("A", result, oxidation)
        [elem_list_a.remove(i) for i in result]
        oxidation = [1 for i in elem_list_a]

        self.set_sites("B", elem_list_a, oxidation)
        self.set_sites("B", b_set, oxidation_states[1])

        if sum(self.result["A"]["fractions"]) != self.ASiteNum:
            raise TypeError("This is not a perovskite or a program recognition error")
        if sum(self.result["B"]["fractions"]) != self.ASiteNum:
            raise TypeError("This is not a perovskite or a program recognition error")
        return

    def parse_elems(self, oxi_state_comp):
        self.elems = {}
        for elem, oxi_state in oxi_state_comp.items():

            if elem in ["F", "Cl", "Br", "I"]:
                self.set_result_value("X", elem, oxi_state)
                self.result["result"] = sum(self.result["X"]["fractions"])
                self.type = 17
                self.result["type"] = PerovskiteEnum.Halide.value

            elif elem in ["S", "O"]:
                self.set_result_value("X", elem, oxi_state)
                self.result["result"] = sum(self.result["X"]["fractions"])
                self.type = 16
                if elem == "S":
                    self.result["type"] = PerovskiteEnum.Sulfide.value
                else:
                    self.result["type"] = PerovskiteEnum.Oxide.value

            else:
                if oxi_state not in self.elems.keys():
                    self.elems[oxi_state] = []
                self.elems[oxi_state].append(Element(elem))

        self.ASiteNum *= self.result["result"] / 3
        if self.type == 17:
            oxidation_states = list(self.elems.keys())
            oxidation_states.sort()
            self.parse_halide_perovskite(oxidation_states)

        elif self.type == 16:
            oxidation_states = list(self.elems.keys())
            oxidation_states.sort()
            self.parse_halide_perovskite(oxidation_states)

        self.PARSE = True

    def get_best_oxi_state(self, comp, all_oxi_states, oxi_states_override, target_charge):

        # Load prior probabilities of oxidation states, used to rank solutions
        if not Composition.oxi_prob:
            module_dir = os.path.join(pymatgen.__path__[0])
            all_data = loadfn(os.path.join(module_dir, "analysis", "icsd_bv.yaml"))
            Composition.oxi_prob = {Species.from_str(sp): data for sp, data in all_data["occurrence"].items()}
        oxi_states_override = oxi_states_override or {}
        # assert: Composition only has integer amounts
        if not all(amt == int(amt) for amt in comp.values()):
            raise ValueError("Charge balance analysis requires integer values in Composition!")

        # for each element, determine all possible sum of oxidations
        # (taking into account nsites for that particular element)
        el_amt = comp.get_el_amt_dict()
        elements = list(el_amt)
        el_sums = []  # matrix: dim1= el_idx, dim2=possible sums
        el_sum_scores = collections.defaultdict(set)  # dict of el_idx, sum -> score
        el_best_oxid_combo = {}  # dict of el_idx, sum -> oxid combo with best score
        for idx, el in enumerate(elements):
            el_sum_scores[idx] = {}
            el_best_oxid_combo[idx] = {}
            el_sums.append([])
            if oxi_states_override.get(el):
                oxids = oxi_states_override[el]
            elif all_oxi_states:
                oxids = Element(el).oxidation_states
            else:
                oxids = Element(el).icsd_oxidation_states or Element(el).oxidation_states
            # print(el, oxids)

            # get all possible combinations of oxidation states
            # and sum each combination
            for oxid_combo in combinations(oxids, 1):
                oxid_combo = [oxid_combo[0] for i in range(int(el_amt[el]))]
                # print(oxid_combo)
                # List this sum as a possible option
                oxid_sum = sum(oxid_combo)
                if oxid_sum not in el_sums[idx]:
                    el_sums[idx].append(oxid_sum)
                    # print(idx,oxid_sum)
                # Determine how probable is this combo?
                score = sum(Composition.oxi_prob.get(Species(el, o), 0) for o in oxid_combo)

                # If it is the most probable combo for a certain sum,
                #   store the combination
                if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(oxid_sum, 0):
                    el_sum_scores[idx][oxid_sum] = score
                    el_best_oxid_combo[idx][oxid_sum] = oxid_combo

        # Determine which combination of oxidation states for each element
        #    is the most probable
        all_sols = []  # will contain all solutions
        all_oxid_combo = []  # will contain the best combination of oxidation states for each site
        all_scores = []  # will contain a score for each solution
        for x in product(*el_sums):
            # each x is a trial of one possible oxidation sum for each element
            if sum(x) == target_charge:  # charge balance condition
                el_sum_sol = dict(zip(elements, x))  # element->oxid_sum
                # normalize oxid_sum by amount to get avg oxid state
                sol = {el: v / el_amt[el] for el, v in el_sum_sol.items()}
                # print(el_sum_sol)
                # add the solution to the list of solutions
                all_sols.append(sol)

                # determine the score for this solution
                score = 0
                for idx, v in enumerate(x):
                    score += el_sum_scores[idx][v]
                all_scores.append(score)

                # collect the combination of oxidation states for each site
                all_oxid_combo.append({e: el_best_oxid_combo[idx][v] for idx, (e, v) in enumerate(zip(elements, x))})

        # sort the solutions by highest to lowest score
        if all_scores:
            all_sols, all_oxid_combo = zip(
                *(
                    (y, x)
                    for (z, y, x) in sorted(
                    zip(all_scores, all_sols, all_oxid_combo),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                )
            )
        return all_sols  # , all_oxid_combo

    def parse_sites(self, all_oxi_states=False):
        if self.PARSE:
            return self.result
        comp, factor = self.compostion.get_integer_formula_and_factor()

        self.oxi_state_guesses = self.get_best_oxi_state(Composition(comp), all_oxi_states, self.oxi_states, 0)

        for oxi_state_comp in self.oxi_state_guesses:
            try:
                self.parse_elems(oxi_state_comp)
                break
            except:
                self.ASiteNum = 1
                for site in ["A", "B", "X"]:
                    self.result[site]["fractions"].clear()
                    self.result[site]["systems"].clear()
                    self.result[site]["valence"].clear()
                self.PARSE = False

    def get_site_radii(self, radiis, weight=None):
        if weight is None:
            weight = [1 / len(radiis) for i in range(len(radiis))]
        else:
            if sum(weight) != 1:
                weight = [i / sum(weight) for i in weight]

        return sum([radiis[i] * weight[i] for i in range(len(radiis))])

    def verify_double_perovskite(self):

        radii_a = self.get_site_radii(
            [self.atoms_source.get_shannon_radii(self.result["A"]["systems"][i], self.result["A"]["valence"][i], 12) for
             i in range(len(self.result["A"]["systems"]))],
            weight=self.result["A"]["fractions"])
        radii_b = self.get_site_radii(
            [self.atoms_source.get_shannon_radii(self.result["B"]["systems"][i], self.result["B"]["valence"][i], 6) for
             i in range(len(self.result["B"]["systems"]))],
            weight=self.result["B"]["fractions"])
        radii_x = self.get_site_radii(
            [self.atoms_source.get_shannon_radii(i, self.result["X"]["valence"][0], 6) for i in
             self.result["X"]["systems"]],
            weight=self.result["X"]["fractions"])

        if radii_a <= radii_b:
            return (-1, -1, -1)

        o = radii_b / radii_x

        t = (radii_a + radii_x) / (math.sqrt(2) * (radii_b + radii_x))
        new_t = radii_x / radii_b - (1 - (radii_a / radii_b) / math.log(radii_a / radii_b, math.e))

        return (round(o, 2), round(t, 2), round(new_t, 2))

    def verify_alloys_perovskite(self):
        total = self.verify_double_perovskite()

        if total[0] <= 0.4 or total[0] >= 1 or total[1] <= 0.82 or total[1] >= 1.08 or total[2] >= 4.18:
            return False
        return True


class MyFeatures(BaseFeaturizer):
    def __init__(self,
                  feature_a=None,
                 feature_b=None,

                 feature_x=None,

                 ):
        self.config={
            "A":[12,feature_a],
            "B":[6,feature_b],
            "X":[6,feature_x]
        }
        self.data_set = set()
        self.DropASiteNum = 0
        self.DropBSiteNum =0
        self.DropB1SiteNum =0
        self.DropXSiteNum =0

        self.atoms_source = AtomsData("./dataset/inorganic_tables.json")
    def get_elem_prop(self,elem:Element,prop,parse_result=None):
        if parse_result is None:
            parse_result=self.parse_result
        if  elem in parse_result["A"]["systems"]:
            index="A"
        elif elem in parse_result["B"]["systems"]:
            index="B"
        elif elem in parse_result["B1"]["systems"]:
            index="B1"
        elif elem in parse_result["X"]["systems"]:
            index="X"
        else:
            raise ValueError("elem not in self.parse_result")

        return index,parse_result[index][prop][parse_result[index]["systems"].index(elem)]



    def featurize(self, comp):
        if isinstance(comp,str):
            comp=Composition(comp)

        features = []
        perovskite = Perovskite(comp)

        if not perovskite.PARSE:
            raise ValueError(comp)
        for elem_index ,elem_index_config in self.config.items():
            species=[Species(perovskite.result[elem_index]["systems"][i],perovskite.result[elem_index]["valence"][i]) for i in range(len(perovskite.result[elem_index]["systems"]))]
            weight=[perovskite.result[elem_index]["fractions"][i]    for i in range(len(perovskite.result[elem_index]["systems"]))]


            feature =self.atoms_source.get_elements_propertys(species=species,weight=weight,coordination=elem_index_config[0])

            if elem_index_config[1] is not None:
                features.extend(feature[elem_index_config[1]])
            else:
                features.extend(feature)
        return features

    def feature_labels(self):
        lables = []
        for elem_index, elem_index_config in self.config.items():
                lable=elem_index_config[1]  if elem_index_config[1] is not None else self.atoms_source.lables
                lables.extend([rf"${i}_{elem_index}$"  for i in lable])
        return lables

    def citations(self):

        citations = []
        return citations

    def implementors(self):
        return ["ccb"]





def gen(a_list=None,
        b1_list=None,
        b2_list=None,
        x_list=None,
        doping_ratio=None):
    for i in product(product(a_list, repeat=len(doping_ratio[0])),
                     product(b1_list, repeat=len(doping_ratio[1])),
                     product(b2_list, repeat=len(doping_ratio[2])),
                     product(x_list, repeat=len(doping_ratio[3]))):

        if set(i[0]) & set(i[1]) or set(i[0]) & set(i[2]):
            continue
        system = ""
        for index, ratios in enumerate(doping_ratio):
            # print(index,ratios)
            for index_, ratio in enumerate(ratios):
                system += i[index][index_] + str(ratio)

        yield system


def verify(system):
    p = Perovskite(system)
    if not p.PARSE:
        return
    try:
        if not p.verify_alloys_perovskite():
            return
    except:

        return
    return Composition(system).formula.replace(" ", "")


def random_perovskites(a_list=None,
                       b1_list=None,
                       b2_list=None,
                       x_list=None,
                       doping_ratio=None,
                       _filter=False
                       ):
    if doping_ratio is None:
        doping_ratio = [[1], [0.5], [0.5], [3]]
    systems = gen(a_list, b1_list, b2_list, x_list, doping_ratio)

    if _filter:
        parallel = Parallel(n_jobs=os.cpu_count(), return_as='generator',
                            verbose=True)  # The return_generator parameter is only supported in py3.7 and later versions.
        output_generator = parallel(delayed(verify)(i) for i in systems)

        for i in output_generator:
            if i:
                yield i
    else:
        for i in systems:
            if i:
                yield Composition(i).formula.replace(" ", "")
