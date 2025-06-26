# Without the internet

1. Get the path:
   ```
   from libsvmdata.datasets import get_data_home
   get_data_home()
   ```
2. Open the file python3.XX/site-packages/libsvmdata/datasets.py, and see where does the dataset goes (**binary** or **multiclass**), and then download it with wget.
2. For example, DNA have to be download in **multiclass**
   