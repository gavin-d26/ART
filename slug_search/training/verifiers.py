import json


def check_answer_correctness_multi_gt(answer: str, correct_answer: str) -> bool:
    correct_answer = json.loads(correct_answer)
    for answer_item in correct_answer:
        if answer_item not in answer:
            return False
    return True
