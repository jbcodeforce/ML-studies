from featureform import Client

client = Client(insecure=True)
dataset = client.training_set("fraud_training","quickstart")

for i, batch in enumerate(dataset):
    print(batch)
    if i > 25:
        break