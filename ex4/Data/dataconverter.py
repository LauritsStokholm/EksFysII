# # # # # # # # Part 0 ~ Converting csv into good format # # # # # # # # # # #
""" BEWARE: ONLY RUN THIS ONCE!
This file is created to change csv formatting """


for item in data_dir:
    # Read only
    with open(item, "r") as file:
        my_file = file.read()
        my_file = my_file.replace(',', '.')  # Changing decimal to american
        my_file = my_file.replace('\t', ',') # Changing delimiter to commas
        my_file = my_file.replace(';', ',')  # Changing delimiter to commas
        my_file = my_file.replace(' ', '')   # Removing spaces
        my_file = my_file.replace(',', ', ') # Adding readable spaces

    # Write only
    with open(item, "w") as file:
        file.write(my_file)
