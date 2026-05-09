from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple, Any, Set

@dataclass
class DUCGNode:
    name: str
    states: List[str]
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    causal_function_table: Dict[Any, Any] = field(default_factory=dict)

@dataclass
class EvidenceFactor:

    x_node: str
    x_state: str
    b_node: str
    a_value: float
    b_prior: float


@dataclass
class EvidenceExpandedTerm:

    factors: List[EvidenceFactor]
    unique_b_nodes: Set[str]
    a_product: float
    prior_product_after_absorption: float
    value_after_absorption: float
    original_form: str
    simplified_form: str


@dataclass
class HEExpandedTerm:

    target_b: str
    source_unique_b_nodes: Set[str]
    absorbed: bool
    evidence_value: float
    target_prior: float
    value_after_absorption: float
    original_form: str
    simplified_form: str


class DUCGFloodRiskModel:
    def __init__(self):
        self.nodes: Dict[str, DUCGNode] = {}

        self._build_nodes()
        self._build_indicator_thresholds()
        self._build_prior_probabilities()
        self._build_allowed_events()
        self._build_x_to_b_causal_table()
        self._build_node_causal_function_tables()


    def _build_nodes(self):

        self.nodes["B1"] = DUCGNode(
            name="B1_高风险",
            states=["connected", "not_connected"],
            parents=[],
            children=["X4", "X5", "X6"]
        )

        self.nodes["B2"] = DUCGNode(
            name="B2_中风险",
            states=["connected", "not_connected"],
            parents=[],
            children=["X4", "X5", "X6"]
        )

        self.nodes["B3"] = DUCGNode(
            name="B3_低风险",
            states=["connected", "not_connected"],
            parents=[],
            children=["X4", "X5", "X6"]
        )

        self.nodes["X4"] = DUCGNode(
            name="X4_暴露性",
            states=["low", "high"],
            parents=["B1", "B2", "B3"],
            children=["elevation", "slope", "river_density", "ndvi"]
        )

        self.nodes["X5"] = DUCGNode(
            name="X5_脆弱性",
            states=["low", "high"],
            parents=["B1", "B2", "B3"],
            children=["population_density", "land_use", "gdp"]
        )

        self.nodes["X6"] = DUCGNode(
            name="X6_危险性",
            states=["low", "high"],
            parents=["B1", "B2", "B3"],
            children=["daily_rainfall", "annual_rainfall"]
        )

        self.nodes["elevation"] = DUCGNode(
            name="高程",
            states=["low", "medium", "high"],
            parents=["X4"],
            children=[]
        )

        self.nodes["slope"] = DUCGNode(
            name="坡度",
            states=["low", "medium", "high"],
            parents=["X4"],
            children=[]
        )

        self.nodes["river_density"] = DUCGNode(
            name="河网密度",
            states=["low", "medium", "high"],
            parents=["X4"],
            children=[]
        )

        self.nodes["ndvi"] = DUCGNode(
            name="NDVI",
            states=["low", "medium", "high"],
            parents=["X4"],
            children=[]
        )

        self.nodes["population_density"] = DUCGNode(
            name="人口密度",
            states=["low", "medium", "high"],
            parents=["X5"],
            children=[]
        )

        self.nodes["land_use"] = DUCGNode(
            name="土地利用类型",
            states=["low", "medium", "high"],
            parents=["X5"],
            children=[]
        )

        self.nodes["gdp"] = DUCGNode(
            name="GDP",
            states=["low", "medium", "high"],
            parents=["X5"],
            children=[]
        )

        self.nodes["daily_rainfall"] = DUCGNode(
            name="日降雨量",
            states=["low", "medium", "high"],
            parents=["X6"],
            children=[]
        )

        self.nodes["annual_rainfall"] = DUCGNode(
            name="年降雨量",
            states=["low", "medium", "high"],
            parents=["X6"],
            children=[]
        )

    def _build_indicator_thresholds(self):

        self.indicator_thresholds: Dict[str, Tuple[float, float]] = {
            # X4 暴露性子变量
            "elevation": (100, 300),
            "slope": (12, 16),
            "river_density": (0.1, 0.3),
            "ndvi": (0.40, 0.70),

            # X5 脆弱性子变量
            "population_density": (500.0, 700.0),
            "land_use": (0.30, 0.60),
            "gdp": (1000.0, 2000.0),

            # X6 危险性子变量
            "daily_rainfall": (25.0, 50.0),
            "annual_rainfall": (1000.0, 2000.0)
        }


    def _build_prior_probabilities(self):

        self.prior_risk: Dict[str, float] = {
            "B1": 0.6,
            "B2": 0.6,
            "B3": 0.7
        }


    def _build_allowed_events(self):

        self.allowed_events: Dict[str, List[Tuple[str, str, str]]] = {
            "B1": [
                ("high", "high", "high"),
                ("high", "low", "high"),
                ("low", "high", "high")
            ],

            "B2": [
                ("high", "high", "low"),
                ("high", "low", "low"),
                ("low", "high", "low"),
                ("low", "low", "high"),
                ("high", "low", "high"),
                ("low", "high", "high")
            ],

            "B3": [
                ("low", "low", "low"),
                ("low", "high", "low"),
                ("high", "low", "low")
            ]
        }


    def _build_x_to_b_causal_table(self):

        self.x_to_b_causal_table: Dict[str, Dict[str, Dict[str, float]]] = {
            "B1": {
                "X4": {
                    "low": 0.2,
                    "high": 0.7
                },
                "X5": {
                    "low": 0.2,
                    "high": 0.5
                },
                "X6": {
                    "low": 0.1,
                    "high": 0.9
                }
            },

            "B2": {
                "X4": {
                    "low": 0.5,
                    "high": 0.5
                },
                "X5": {
                    "low": 0.5,
                    "high": 0.5
                },
                "X6": {
                    "low": 0.4,
                    "high": 0.6
                }
            },

            "B3": {
                "X4": {
                    "low": 0.9,
                    "high": 0.1
                },
                "X5": {
                    "low": 0.8,
                    "high": 0.2
                },
                "X6": {
                    "low": 0.9,
                    "high": 0.1
                }
            }
        }


    def _build_node_causal_function_tables(self):

        for child in ["elevation", "slope", "river_density", "ndvi"]:
            self.nodes[child].causal_function_table = {
                ("X4", "low"): {"low": 0.70, "medium": 0.20, "high": 0.10},
                ("X4", "high"): {"low": 0.10, "medium": 0.20, "high": 0.70}
            }

        for child in ["population_density", "land_use", "gdp"]:
            self.nodes[child].causal_function_table = {
                ("X5", "low"): {"low": 0.70, "medium": 0.20, "high": 0.10},
                ("X5", "high"): {"low": 0.10, "medium": 0.20, "high": 0.70}
            }

        for child in ["daily_rainfall", "annual_rainfall"]:
            self.nodes[child].causal_function_table = {
                ("X6", "low"): {"low": 0.7, "medium": 0.20, "high": 0.1},
                ("X6", "high"): {"low": 0.1, "medium": 0.2, "high": 0.70}
            }


    def classify_indicator_state(self, indicator_name: str, value: float) -> str:
        if indicator_name not in self.indicator_thresholds:
            raise KeyError(f"指标 {indicator_name} 未设置阈值。")

        low_threshold, high_threshold = self.indicator_thresholds[indicator_name]

        if value < low_threshold:
            return "low"
        elif value < high_threshold:
            return "medium"
        else:
            return "high"

    def classify_all_indicators(self, sample: Dict[str, float]) -> Dict[str, str]:
        indicator_states: Dict[str, str] = {}

        for indicator in self.indicator_thresholds:
            if indicator not in sample:
                raise KeyError(f"输入样本缺少指标：{indicator}")

            indicator_states[indicator] = self.classify_indicator_state(
                indicator_name=indicator,
                value=sample[indicator]
            )

        return indicator_states

    @staticmethod
    def is_medium_or_high(state: str) -> bool:
        return state in ["medium", "high"]

    def infer_x4_state(self, indicator_states: Dict[str, str]) -> str:

        x4_children = ["elevation", "slope", "river_density", "ndvi"]

        count = 0
        for child in x4_children:
            if self.is_medium_or_high(indicator_states[child]):
                count += 1

        if count >= 3:
            return "high"
        else:
            return "low"

    def infer_x5_state(self, indicator_states: Dict[str, str]) -> str:


        x5_children = ["population_density", "land_use", "gdp"]

        for child in x5_children:
            if not self.is_medium_or_high(indicator_states[child]):
                return "low"

        return "high"

    def infer_x6_state(self, indicator_states: Dict[str, str]) -> str:

        x6_children = ["daily_rainfall", "annual_rainfall"]

        for child in x6_children:
            if not self.is_medium_or_high(indicator_states[child]):
                return "low"

        return "high"

    def infer_x_states(self, indicator_states: Dict[str, str]) -> Dict[str, str]:
        return {
            "X4": self.infer_x4_state(indicator_states),
            "X5": self.infer_x5_state(indicator_states),
            "X6": self.infer_x6_state(indicator_states)
        }


    def get_x_combination(self, x_states: Dict[str, str]) -> Tuple[str, str, str]:
        return (
            x_states["X4"],
            x_states["X5"],
            x_states["X6"]
        )

    def get_b_connection_state(
        self,
        b_node: str,
        x_combination: Tuple[str, str, str]
    ) -> str:
        if x_combination in self.allowed_events[b_node]:
            return "connected"
        else:
            return "not_connected"

    def get_connected_b_nodes(self, x_states: Dict[str, str]) -> List[str]:

        x_combination = self.get_x_combination(x_states)

        connected_b_nodes: List[str] = []

        for b_node in ["B1", "B2", "B3"]:
            if x_combination in self.allowed_events[b_node]:
                connected_b_nodes.append(b_node)

        return connected_b_nodes


    def build_evidence_factors_by_x(
        self,
        x_states: Dict[str, str],
        connected_b_nodes: List[str]
    ) -> Dict[str, List[EvidenceFactor]]:

        factors_by_x: Dict[str, List[EvidenceFactor]] = {}

        for x_node in ["X4", "X5", "X6"]:
            x_state = x_states[x_node]
            factors_by_x[x_node] = []

            for b_node in connected_b_nodes:
                a_value = self.x_to_b_causal_table[b_node][x_node][x_state]
                b_prior = self.prior_risk[b_node]

                factors_by_x[x_node].append(
                    EvidenceFactor(
                        x_node=x_node,
                        x_state=x_state,
                        b_node=b_node,
                        a_value=a_value,
                        b_prior=b_prior
                    )
                )

        return factors_by_x


    def calculate_prior_product_after_absorption(self, b_nodes: Set[str]) -> float:

        prior_product = 1.0

        for b_node in sorted(b_nodes):
            prior_product *= self.prior_risk[b_node]

        return prior_product

    def calculate_pr_e(
        self,
        x_states: Dict[str, str],
        connected_b_nodes: List[str]
    ) -> Tuple[
        Dict[str, List[EvidenceFactor]],
        List[EvidenceExpandedTerm],
        float
    ]:

        factors_by_x = self.build_evidence_factors_by_x(
            x_states=x_states,
            connected_b_nodes=connected_b_nodes
        )

        x4_factors = factors_by_x["X4"]
        x5_factors = factors_by_x["X5"]
        x6_factors = factors_by_x["X6"]

        expanded_terms: List[EvidenceExpandedTerm] = []
        pr_e_total = 0.0

        for factor_tuple in product(x4_factors, x5_factors, x6_factors):
            factors = list(factor_tuple)

            a_product = 1.0
            b_nodes: Set[str] = set()

            original_parts = []
            simplified_a_parts = []

            for factor in factors:
                a_product *= factor.a_value
                b_nodes.add(factor.b_node)

                original_parts.append(
                    f"A({factor.x_node}={factor.x_state}->{factor.b_node})"
                    f"P({factor.b_node})"
                )

                simplified_a_parts.append(
                    f"A({factor.x_node}={factor.x_state}->{factor.b_node})"
                )

            prior_product = self.calculate_prior_product_after_absorption(b_nodes)

            value = a_product * prior_product

            prior_part = "".join([f"P({b})" for b in sorted(b_nodes)])

            original_form = " × ".join(original_parts)

            simplified_form = (
                " × ".join(simplified_a_parts)
                + " × "
                + prior_part
            )

            expanded_term = EvidenceExpandedTerm(
                factors=factors,
                unique_b_nodes=b_nodes,
                a_product=a_product,
                prior_product_after_absorption=prior_product,
                value_after_absorption=value,
                original_form=original_form,
                simplified_form=simplified_form
            )

            expanded_terms.append(expanded_term)
            pr_e_total += value

        return factors_by_x, expanded_terms, pr_e_total

    # ========================================================
    # 12. 计算 Pr{H,E}
    # ========================================================

    def calculate_pr_h_e(
        self,
        target_b: str,
        pr_e_expanded_terms: List[EvidenceExpandedTerm]
    ) -> Tuple[float, List[HEExpandedTerm]]:

        target_prior = self.prior_risk[target_b]

        he_terms: List[HEExpandedTerm] = []
        pr_h_e_total = 0.0

        for e_term in pr_e_expanded_terms:
            if target_b in e_term.unique_b_nodes:
                absorbed = True
                value = e_term.value_after_absorption

                simplified_form = e_term.simplified_form

            else:
                absorbed = False
                value = e_term.value_after_absorption * target_prior

                simplified_form = (
                    e_term.simplified_form
                    + f" × P({target_b})"
                )

            original_form = (
                f"P({target_b}) × ["
                + e_term.original_form
                + "]"
            )

            he_term = HEExpandedTerm(
                target_b=target_b,
                source_unique_b_nodes=set(e_term.unique_b_nodes),
                absorbed=absorbed,
                evidence_value=e_term.value_after_absorption,
                target_prior=target_prior,
                value_after_absorption=value,
                original_form=original_form,
                simplified_form=simplified_form
            )

            he_terms.append(he_term)
            pr_h_e_total += value

        return pr_h_e_total, he_terms


    def infer_risk_posterior(
        self,
        x_states: Dict[str, str]
    ) -> Dict[str, Any]:

        connected_b_nodes = self.get_connected_b_nodes(x_states)

        if not connected_b_nodes:
            raise ValueError(
                "当前 X4、X5、X6 状态组合没有连接任何 B 节点，请检查 allowed_events。"
            )

        factors_by_x, pr_e_expanded_terms, pr_e_total = self.calculate_pr_e(
            x_states=x_states,
            connected_b_nodes=connected_b_nodes
        )

        if pr_e_total <= 0:
            raise ValueError("Pr{E} 为 0，无法计算。")

        pr_h_e_by_b: Dict[str, float] = {
            "B1": 0.0,
            "B2": 0.0,
            "B3": 0.0
        }

        he_terms_by_b: Dict[str, List[HEExpandedTerm]] = {
            "B1": [],
            "B2": [],
            "B3": []
        }

        score_by_b: Dict[str, float] = {
            "B1": 0.0,
            "B2": 0.0,
            "B3": 0.0
        }

        for b_node in connected_b_nodes:
            pr_h_e, he_terms = self.calculate_pr_h_e(
                target_b=b_node,
                pr_e_expanded_terms=pr_e_expanded_terms
            )

            pr_h_e_by_b[b_node] = pr_h_e
            he_terms_by_b[b_node] = he_terms
            score_by_b[b_node] = pr_h_e / pr_e_total

        score_total = sum(score_by_b[b] for b in connected_b_nodes)

        if score_total <= 0:
            raise ValueError("connected B 的 score 总和为 0，无法归一化。")

        posterior: Dict[str, float] = {
            "B1": 0.0,
            "B2": 0.0,
            "B3": 0.0
        }

        for b_node in connected_b_nodes:
            posterior[b_node] = score_by_b[b_node] / score_total

        return {
            "connected_b_nodes": connected_b_nodes,
            "factors_by_x": factors_by_x,
            "pr_e_expanded_terms": pr_e_expanded_terms,
            "pr_e_total": pr_e_total,
            "pr_h_e_by_b": pr_h_e_by_b,
            "he_terms_by_b": he_terms_by_b,
            "score_by_b": score_by_b,
            "posterior": posterior
        }


    def predict(self, sample: Dict[str, float]) -> Dict[str, Any]:
        indicator_states = self.classify_all_indicators(sample)
        x_states = self.infer_x_states(indicator_states)
        x_combination = self.get_x_combination(x_states)

        b_connection_states = {
            b_node: self.get_b_connection_state(
                b_node=b_node,
                x_combination=x_combination
            )
            for b_node in ["B1", "B2", "B3"]
        }

        posterior_result = self.infer_risk_posterior(x_states)

        posterior = posterior_result["posterior"]

        final_b_node = max(posterior, key=posterior.get)

        risk_name_map = {
            "B1": "高风险",
            "B2": "中风险",
            "B3": "低风险"
        }

        return {
            "indicator_states": indicator_states,
            "x_states": x_states,
            "x_combination": x_combination,
            "b_connection_states": b_connection_states,
            "connected_b_nodes": posterior_result["connected_b_nodes"],
            "factors_by_x": posterior_result["factors_by_x"],
            "pr_e_expanded_terms": posterior_result["pr_e_expanded_terms"],
            "pr_e_total": posterior_result["pr_e_total"],
            "pr_h_e_by_b": posterior_result["pr_h_e_by_b"],
            "he_terms_by_b": posterior_result["he_terms_by_b"],
            "score_by_b": posterior_result["score_by_b"],
            "posterior": posterior,
            "final_b_node": final_b_node,
            "final_risk": risk_name_map[final_b_node]
        }

    def print_prediction_result(self, result: Dict[str, Any]):
        risk_name_map = {
            "B1": "高风险",
            "B2": "中风险",
            "B3": "低风险"
        }

        print("\n" + "=" * 80)
        print("DUCG 推理结果")
        print("=" * 80)

        print("\n一、第三层指标状态")
        for indicator, state in result["indicator_states"].items():
            print(f"{indicator}: {state}")

        print("\n二、第二层 X 节点状态")
        for x_node, state in result["x_states"].items():
            print(f"{x_node}: {state}")

        print("\n三、X4、X5、X6 状态组合")
        print(result["x_combination"])

        print("\n四、B1、B2、B3 连接状态")
        for b_node, state in result["b_connection_states"].items():
            print(f"{b_node}: {state}")

        print("\n五、实际参与计算的 B 节点")
        print(result["connected_b_nodes"])

        print("\n六、Pr{E} 的三个括号")
        for x_node in ["X4", "X5", "X6"]:
            factors = result["factors_by_x"][x_node]
            parts = []
            for factor in factors:
                parts.append(
                    f"A({factor.x_node}={factor.x_state}->{factor.b_node})"
                    f"P({factor.b_node})"
                    f"={factor.a_value:.6f}×{factor.b_prior:.6f}"
                )
            print(f"{x_node}: " + " + ".join(parts))

        print("\n七、Pr{E} 展开项及相同 B 先验吸收")
        for idx, term in enumerate(result["pr_e_expanded_terms"], start=1):
            print(f"\n第 {idx} 项:")
            print(f"  原始项: {term.original_form}")
            print(f"  吸收后: {term.simplified_form}")
            print(f"  A乘积 = {term.a_product:.6f}")
            print(f"  先验吸收后乘积 = {term.prior_product_after_absorption:.6f}")
            print(f"  项值 = {term.value_after_absorption:.6f}")

        print(f"\nPr{{E}} = {result['pr_e_total']:.6f}")

        print("\n八、Pr{H,E} 展开与吸收")
        for b_node in ["B1", "B2", "B3"]:
            if b_node not in result["connected_b_nodes"]:
                print(f"\n{b_node}({risk_name_map[b_node]}): not_connected，不参与计算")
                continue

            print(f"\n{b_node}({risk_name_map[b_node]}):")
            print(f"Pr{{H,E}}({b_node}) = P({b_node}) × Pr{{E}}")

            for idx, term in enumerate(result["he_terms_by_b"][b_node], start=1):
                print(f"\n  第 {idx} 项:")
                print(f"    原始项: {term.original_form}")

                if term.absorbed:
                    print(f"    处理: {b_node} 已在该项中出现，先验融合吸收")
                else:
                    print(f"    处理: {b_node} 未在该项中出现，额外乘 P({b_node})")

                print(f"    化简后: {term.simplified_form}")
                print(f"    Pr{{E}}项值 = {term.evidence_value:.6f}")
                print(f"    P({b_node}) = {term.target_prior:.6f}")
                print(f"    Pr{{H,E}}项值 = {term.value_after_absorption:.6f}")

            print(f"\n  Pr{{H,E}}({b_node}) = {result['pr_h_e_by_b'][b_node]:.6f}")

        print("\n九、score = Pr{H,E} / Pr{E}")
        for b_node in ["B1", "B2", "B3"]:
            if b_node not in result["connected_b_nodes"]:
                print(f"{b_node}({risk_name_map[b_node]}): 0.000000（not_connected）")
            else:
                print(
                    f"{b_node}({risk_name_map[b_node]}): "
                    f"{result['score_by_b'][b_node]:.6f}"
                )

        print("\n十、最终归一化后验概率")
        for b_node in ["B1", "B2", "B3"]:
            if b_node not in result["connected_b_nodes"]:
                print(f"{b_node}({risk_name_map[b_node]}): 0.000000（not_connected）")
            else:
                print(
                    f"{b_node}({risk_name_map[b_node]}): "
                    f"{result['posterior'][b_node]:.6f}"
                )

        # print("\n十一、最终风险等级")
        # print(f"{result['final_b_node']}：{result['final_risk']}")


if __name__ == "__main__":
    model = DUCGFloodRiskModel()


    sample = {

        "elevation": 105,
        "slope": 10.02,
        "river_density": 0.41,
        "ndvi": 0.69,


        "population_density": 871,
        "land_use": 0.1,
        "gdp": 2317.47,


        "daily_rainfall": 40.89,
        "annual_rainfall": 2202.1
    }

    result = model.predict(sample)

    model.print_prediction_result(result)