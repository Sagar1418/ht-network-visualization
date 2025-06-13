# HT Network Feeder-wise Visualization

A Streamlit web app to visualize High Tension (HT) cable network data feeder-wise, with advanced multi-edge and network clarity features.

---

## Features

- **CSV Upload:** Easily upload your `HTCABLE.csv` data file.
- **Filter and Multi-select:** Filter data by network type (`CABLE_TYPE`) and feeder ID (`FEEDER_ID`). Supports multiple selection.
- **Smart Graph Visualization:**  
  - Visualizes the HT cable network as a graph using PyVis.
  - **Node size is large and displays the count of total connections (incoming + outgoing)** on each node label.
  - Multiple edges between the same nodes are colored differently and highlighted.
- **Interactive Table:**  
  - See only the multi-edge (duplicate edge) connections with custom columns.
  - Color legend in the table matches the graph edge colors.
- **Comment/Remark Search:**  
  - Search and filter connections by any word in the comments.
- **Handles encoding errors** in CSV files automatically.
- **Fast, easy, and highly customizable!**

---

## How To Use

1. **Clone this repository or copy the code files.**
2. **Install requirements (preferably in a virtual environment):**

    ```bash
    pip install streamlit pandas pyvis networkx
    ```

3. **Run the Streamlit app:**

    ```bash
    streamlit run visual.py
    ```

4. **Upload your `HTCABLE.csv` file** when prompted in the app UI.

---

## CSV File Format

Your CSV should have at least these columns:
- `FEEDERID`
- `SOURCE_SS`
- `DESTINATION_SS`
- `LABELTEXT`
- `MEASUREDLENGTH`
- `COMMENTS`
- `REMARK` (optional but recommended)

**If your columns have different names, rename them or adjust the code accordingly.**

---

## Graph Display Details

- **Each node** label shows the name and the total number of connections, e.g. `KURLA REC-STN (5)`.
- **Edges**: Multiple connections between same nodes are highlighted with different colors and thicker lines.
- **Node size**: All nodes are set to a large size for easy viewing.

---

## Customization

- You can select which columns to show in the explanation table.
- You can search/filter data directly in the app.
- The physics and layout of the network graph can be changed by editing the physics options in the code.

---


