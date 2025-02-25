import os
import operator
import pandas as pd
from functools import reduce
from utils import answer_is_correct
from prompt_tools import LLMConfig



class CSVLogger:
    def __init__(self, file_path):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
        pd.options.mode.copy_on_write = True
        self.file_path = file_path
        self.false_agents = ["question", "answer", "gqa_question", "asp_question", "gqa", "asp"]
        if os.path.exists(file_path):
            self.records = pd.read_csv(file_path, dtype={"qid": str, "answer": str})
        else:
            self.records = pd.DataFrame(columns=["qid", "agent", "answer", *LLMConfig.ablation_categories(), "nr_actual_examples"])
        self.numeric_columns = ["nr_actual_examples", "n_similar_examples"]
        for col in self.numeric_columns:
            self.records[col] = pd.to_numeric(self.records[col], errors="coerce")

    def log(self, qid, agent, answer, n_actual_examples=None, raw=False, config: LLMConfig = None):
        agent = f"{agent}_raw" if raw else agent
        n_actual_examples = n_actual_examples if n_actual_examples is not None else "-"
        ablation_values = config.get_ablation_values() if config is not None else ["-"]*len(LLMConfig.ablation_categories())
        self.records.loc[len(self.records.index)] = [str(qid), agent, answer, *ablation_values, n_actual_examples]
        
    def log_safe(self, qid, agent, answer, nr_actual_examples=None, raw=None, config: LLMConfig = None):
        if not self.is_answered(qid, agent, config):
            self.log(qid, agent, answer, nr_actual_examples, raw, config)

    def get_answer(self, qid, agent, config: LLMConfig = None):
        conditions = [self.records["qid"] == qid, self.records["agent"] == agent]
        if config is not None:
            for category, val in zip(LLMConfig.ablation_categories(), config.get_ablation_values()):
                if category in self.numeric_columns:
                    conditions.append(self.records[category] == float(val))
                else:
                    conditions.append(self.records[category] == str(val))
        return self.records[reduce(operator.and_, conditions)]["answer"]

    def is_answered(self, qid, agent, config: LLMConfig = None):
        return len(self.get_answer(qid, agent, config)) > 0

    def save(self):
        self.records.to_csv(self.file_path, index=False)
    
    def models_present(self):
        return [agent for agent in self.records["agent"].unique() if not (agent in self.false_agents or agent.endswith("_raw"))]
    
    def model_answers(self, model, raw=False):
        if raw:
            answers = self.records[self.records["agent"] == f"{model}_raw"]
        else:
            answers = self.records[self.records["agent"] == model]
        return pd.merge(answers, self.records[self.records["agent"] == "answer"][["qid", "answer"]], on="qid", suffixes=("_pred", "_gt"))
    
    def check_correct(self, answers, use_wordnet=False, wordnet_hypernym_levels=0, topk=1):
        def _row_correct(row):
            gt_answer = self.records[(self.records["agent"] == "answer") & (self.records["qid"] == row["qid"])]["answer"].values[0]
            return answer_is_correct(row["answer"], gt_answer, use_wordnet=use_wordnet, hypernym_levels=wordnet_hypernym_levels, top_k=topk)
        return answers.apply(_row_correct, axis=1)
    
    def answerable_questions(self, authority: str = "asp", use_wordnet=False, wordnet_hypernym_levels=0, topk=1):
        authority_answers = self.records[self.records["agent"] == authority]
        return authority_answers[self.check_correct(authority_answers,use_wordnet=use_wordnet, wordnet_hypernym_levels=wordnet_hypernym_levels, topk=topk)]["qid"]
    
    def score(self, categories="all", base_agents=False, filter_by_answerable_authority=None, use_wordnet=False, wordnet_hypernym_levels=0, topk=1):
        agents_of_interest = self.models_present() if not base_agents else ["gqa", "asp"]
        answers = self.records[self.records["agent"].isin(agents_of_interest)]
        if filter_by_answerable_authority is not None:
            # pray that this caches the answerable questions
            answers = answers[answers["qid"].isin(self.answerable_questions(filter_by_answerable_authority, use_wordnet=use_wordnet, wordnet_hypernym_levels=wordnet_hypernym_levels, topk=topk))]
        answers["correct"] = self.check_correct(answers, use_wordnet, wordnet_hypernym_levels, topk=topk)
        if categories == "all":
            categories = LLMConfig.ablation_categories()
        score = answers.groupby(["agent", *categories]).agg(Accuracy=("correct", "mean"), n=("correct", "count"), mean_examples=("nr_actual_examples", "mean"))
        score.rename(columns={"n": "#Questions", "agent": "Model"}, inplace=True)
        for cat in categories:
            score.rename(columns={cat: LLMConfig.cat_to_human_readable(cat)}, inplace=True)
        return score
    
    def drop_agent(self, agent):
        self.records = self.records[self.records["agent"] != agent]
        self.records = self.records[self.records["agent"] != agent+"_raw"]
        self.save()
        
        
    

if __name__ == "__main__":
    logger = CSVLogger("./logs/ablationlog.csv")
    print(logger.score())
    logger.score(categories=["target_repr"]).to_latex(escape=True)
    pass
