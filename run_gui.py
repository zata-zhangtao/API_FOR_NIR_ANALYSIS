import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import datetime
import json
import os
from typing import Union
import pymysql

# 假设这是你的函数集合（这里仅示例部分函数，实际使用时替换为你的完整代码）
def transform_xlsx_to_mysql(file_path, machine_type="卷积式_v1", upload_database=False, **kw):
    # 这里仅模拟函数行为，实际使用你的实现
    return f"Transformed {file_path} to MySQL with machine_type={machine_type}, upload_database={upload_database}"

def create_connection_for_Guangyin_database(database, host="192.168.110.150", port=53306, user="root", password="Guangyin88888888@", charset="utf8mb4"):
    return f"Connected to Guangyin database: {database}"

def insert_spectrum_data_to_mysql(table_name, 光谱, 项目名称, 项目类型, 采集日期, 理化值, 创建时间, **kwargs):
    return f"Inserted spectrum data into {table_name}"

# 函数字典，方便扩展
FUNCTIONS = {
    "Database Operations": {
        "transform_xlsx_to_mysql": {
            "func": transform_xlsx_to_mysql,
            "params": [
                ("file_path", str, "File Path", ""),
                ("machine_type", str, "Machine Type", "卷积式_v1"),
                ("upload_database", bool, "Upload to Database", False)
            ]
        },
        "create_connection_for_Guangyin_database": {
            "func": create_connection_for_Guangyin_database,
            "params": [
                ("database", str, "Database Name", ""),
                ("host", str, "Host", "192.168.110.150"),
                ("port", int, "Port", 53306),
                ("user", str, "User", "root"),
                ("password", str, "Password", "Guangyin88888888@"),
                ("charset", str, "Charset", "utf8mb4")
            ]
        },
        "insert_spectrum_data_to_mysql": {
            "func": insert_spectrum_data_to_mysql,
            "params": [
                ("table_name", str, "Table Name", ""),
                ("光谱", list, "Spectrum Data (as list)", "[]"),
                ("项目名称", str, "Project Name", ""),
                ("项目类型", str, "Project Type", ""),
                ("采集日期", str, "Collection Date", ""),
                ("理化值", dict, "Physical-Chemical Values (as dict)", "{}"),
                ("创建时间", str, "Creation Time", "")
            ]
        }
    },
    # 你可以添加其他类别，例如 "Data Processing", "Model Operations" 等
    "Data Processing": {
        # 示例占位函数
        "sort_by_datetime": {
            "func": lambda datetime_array, *data_arrays: "Sorted by datetime",
            "params": [
                ("datetime_array", str, "Datetime Array", ""),
                ("data_arrays", str, "Data Arrays (comma-separated)", "")
            ]
        }
    }
}

class FunctionCallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Function Caller GUI")
        self.root.geometry("800x600")

        # 主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # 函数类别选择
        ttk.Label(self.main_frame, text="Select Function Category:").grid(row=0, column=0, sticky="w")
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(self.main_frame, textvariable=self.category_var, 
                                          values=list(FUNCTIONS.keys()))
        self.category_combo.grid(row=0, column=1, sticky="w")
        self.category_combo.bind("<<ComboboxSelected>>", self.update_function_list)

        # 函数选择
        ttk.Label(self.main_frame, text="Select Function:").grid(row=1, column=0, sticky="w")
        self.function_var = tk.StringVar()
        self.function_combo = ttk.Combobox(self.main_frame, textvariable=self.function_var)
        self.function_combo.grid(row=1, column=1, sticky="w")
        self.function_combo.bind("<<ComboboxSelected>>", self.update_param_fields)

        # 参数输入区域
        self.param_frame = ttk.LabelFrame(self.main_frame, text="Parameters", padding="5")
        self.param_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.param_entries = {}

        # 执行按钮
        self.run_button = ttk.Button(self.main_frame, text="Run Function", command=self.run_function)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=10)

        # 结果显示
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Result", padding="5")
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.result_text = scrolledtext.ScrolledText(self.result_frame, height=15, width=80)
        self.result_text.grid(row=0, column=0)

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=1)

    def update_function_list(self, event):
        category = self.category_var.get()
        if category:
            functions = list(FUNCTIONS[category].keys())
            self.function_combo["values"] = functions
            self.function_combo.set("")
            self.clear_param_fields()

    def update_param_fields(self, event):
        self.clear_param_fields()
        category = self.category_var.get()
        function_name = self.function_var.get()
        if category and function_name:
            params = FUNCTIONS[category][function_name]["params"]
            for i, (param_name, param_type, label, default) in enumerate(params):
                ttk.Label(self.param_frame, text=f"{label} ({param_type.__name__}):").grid(row=i, column=0, sticky="w")
                entry = ttk.Entry(self.param_frame)
                entry.insert(0, str(default))
                entry.grid(row=i, column=1, sticky="ew")
                self.param_entries[param_name] = (entry, param_type)

    def clear_param_fields(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

    def run_function(self):
        category = self.category_var.get()
        function_name = self.function_var.get()
        if not category or not function_name:
            messagebox.showwarning("Warning", "Please select a category and function.")
            return

        func_info = FUNCTIONS[category][function_name]
        func = func_info["func"]
        params = {}
        
        try:
            for param_name, (entry, param_type) in self.param_entries.items():
                value = entry.get()
                if param_type == bool:
                    params[param_name] = value.lower() in ("true", "1", "yes")
                elif param_type == int:
                    params[param_name] = int(value)
                elif param_type == list:
                    params[param_name] = json.loads(value)  # 假设输入是 JSON 格式的列表
                elif param_type == dict:
                    params[param_name] = json.loads(value)  # 假设输入是 JSON 格式的字典
                else:
                    params[param_name] = value

            result = func(**params)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, str(result))
        except Exception as e:
            messagebox.showerror("Error", f"Error executing function: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FunctionCallerGUI(root)
    root.mainloop()