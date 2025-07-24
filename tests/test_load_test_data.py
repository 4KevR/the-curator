from src.backend.modules.evaluation.load_test_data.load_test_data import load_test_data

test_data_path = "../tests/data/tests.json"
test_data = load_test_data(test_data_path)

print(test_data)
