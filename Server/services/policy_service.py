from services.mongo import get_collection

async def get_all_policies():
    policies = get_collection("policies")
    return await policies.find().to_list(100)

async def get_policy_by_id(policy_id: str):
    policies = get_collection("policies")
    return await policies.find_one({"_id": policy_id})
