import openml
import pandas as pd

def fetch_datasets_less_than_1000_samples():
    """
    Return a list of (dataset_id, name, is_classification) for OpenML datasets
    with < 1000 instances. We try to guess classification vs. regression.
    """

    # Query OpenML for datasets with <1000 samples
    # Using 'number_of_instances < 1000' filter
    dataset_list = openml.datasets.list_datasets(
        size=10000,  # large upper bound to list many datasets
        status='active'
    )
    
    # Convert dict -> DataFrame for easier filtering
    df = pd.DataFrame.from_dict(dataset_list, orient='index')
    df = df[df['NumberOfInstances'] < 1000]

    results = []
    for did in df['did'].values:
        try:
            dataset = openml.datasets.get_dataset(did)
            # Attempt basic classification/regression check
            # We'll require a default target to proceed
            target_name = dataset.default_target_attribute
            if target_name is None:
                continue

            X, y, categorical, attribute_names = dataset.get_data(
                target=target_name, dataset_format="dataframe"
            )
            
            # Simple heuristic:
            # If y is numeric (float or int with many unique values), treat as regression
            # If y has few unique values or is of type 'object'/'category', treat as classification
            if pd.api.types.is_numeric_dtype(y):
                unique_vals = y.nunique()
                if unique_vals <= 20:
                    # Probably classification
                    is_classification = True
                else:
                    # Probably regression
                    is_classification = False
            else:
                # text/categorical => classification
                is_classification = True

            # Return dataset ID, name, and classification/regression
            results.append((did, dataset.name, is_classification))

        except Exception as e:
            # If anything goes wrong, just skip
            continue

    return results

if __name__ == "__main__":
    datasets_info = fetch_datasets_less_than_1000_samples()
    print("Found {} datasets with < 1000 samples.".format(len(datasets_info)))
    for d in datasets_info[:10]:
        print(d)
