import pandas as pd

df = pd.read_csv("FINALDUMMY.csv", index_col=0)


def df_to_xml(df):
    xml_by_class = {}
    sort_vehicle_class = [i for _, i in df.groupby('class')]
    for vehicle_df in sort_vehicle_class:
        sort_time = [i for i in vehicle_df.groupby('Begin')]
        body_message = []
        for sample in sort_time:
            sample_df = sample[1]
            sample_df = sample_df.reset_index(drop=True)
            starting_message=' <interval id="begin_{}" begin="{}" end="{}">\n'.format(sample_df['Begin'][0],sample_df['Begin'][0],sample_df['End'][0])
            body_message.append(starting_message)
            for idx in range(sample_df.shape[0]):
                _,_,vehicle_type,count,from_edge,to_edge= sample_df.iloc[idx,:]
                if count == 0:
                    continue
                message = '  <edgeRelation from="{}" to="{}" count="{}"/>\n'.format(from_edge,to_edge,count)
                body_message.append(message)
        body_message.append(' </interval>\n')
        xml_by_class[vehicle_type] = body_message

    for key, val in xml_by_class.items():
        filename = f"{key}.xml"
        start = "<data>\n"
        end = "</data>"

        with open(f"Route Files\\{filename}", "w") as f:
            f.write(start)
            for line in val:
                f.write(line)
            f.write(end)
    print("Sorted Done")
df_to_xml(df)