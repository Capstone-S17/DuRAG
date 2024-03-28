import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()
collection = client.collections.get("SWR_chunks")

target_id = "8a8be102-6b24-57c8-89ae-38c1a94aa051"
response = collection.query.fetch_objects(
    filters=wvc.query.Filter.by_id().equal(target_id)
)

for o in response.objects:
    print(o.properties)  # Inspect returned objects
    print(o.uuid)
    print("\n\n\n")

response = collection.query.fetch_objects(
    filters=(
        wvc.query.Filter.by_property("pdf_name").equal(
            "SG230907OTHR98Q9_Totm Technologies Limited_20230907225023_00_AR_4Q_20230531.1.pdf"
        )
        | wvc.query.Filter.by_property("pdf_name").equal(
            "SG230825OTHRWBL2_Wesfarmers Limited_20230825060128_00_GA_4Q_20230825.1.pdf"
        )
        | wvc.query.Filter.by_property("pdf_name").equal(
            "SG210628OTHRHZH9_Duty Free International Limited_20210628175748_00_GA_4Q_20210701.1.pdf"
        )
        | wvc.query.Filter.by_property("pdf_name").equal(
            "SG220831OTHRXDF6_Blackrock (Singapore) Limited_20220831171053_00_GA_4Q_20220831.1.pdf"
        )
    ),
    limit=3,
)

for o in response.objects:
    print(o.properties)  # Inspect returned objects
    print(o.uuid)
