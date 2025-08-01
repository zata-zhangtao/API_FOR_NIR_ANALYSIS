import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import io
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import os
matplotlib.use('Agg')
import scipy
import sys
import random
import warnings

class SpectralAnalysisReport:
    def __init__(self, dataset, output_path='spectral_analysis_report.pdf'):
        """
        Initialize spectral data analysis report class
        """


        
        # Clear matplotlib font cache and set default fonts
        import matplotlib.font_manager as fm
        try:
            # Force reload font manager without cache
            fm._load_fontmanager(try_read_cache=False)
        except:
            pass
        
        # # Explicitly set matplotlib to use default fonts only
        # plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # # Disable any Chinese font fallback
        # plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
        # plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Courier', 'monospace']
        
        # Set logging level to suppress font warnings
        import logging
        import warnings
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        logging.getLogger('matplotlib.fontmanager').setLevel(logging.ERROR) 
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', message='.*font.*')
        warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
        
        # Check if dataset contains required spectral data
        if 'spectra' not in dataset:
            raise KeyError("Dataset must contain 'spectra' key")

        if 'measured_value' not in dataset:
            print("\033[91m⚠️  WARNING: If Dataset contain 'measured_value' key, it will be more useful\033[0m")
        if 'collection_date' not in dataset:
            print("\033[93m⚠️  WARNING: If Dataset contain 'collection_date' key, it will be more useful\033[0m")
        if 'volunteer' not in dataset:
            print("\033[96m⚠️  WARNING: If Dataset contain 'volunteer' key, it will be more useful\033[0m")
            
        self.dataset = dataset
        self.output_path = output_path
        self.spectral_data = dataset['spectra']
        self.n_samples, self.n_features = self.spectral_data.shape
        
        # Initialize PDF document
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.pdf_elements = []
        # self.analyze_and_generate_report()

    def _setup_fonts(self):
        """Setup fonts for PDF generation"""
        # Use default Helvetica font for PDF
        self.font_available = True

    def _setup_styles(self):
        """Setup document styles"""
        # Use default Helvetica font
        font_name = 'Helvetica'
        
        # Check if custom styles already exist, if not add them
        if 'CustomHeading1' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomHeading1',
                fontName=font_name,
                fontSize=18,
                leading=22,
                spaceAfter=12,
                alignment=1  # Center
            ))
        
        if 'CustomHeading2' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomHeading2',
                fontName=font_name,
                fontSize=16,
                leading=20,
                spaceAfter=10,
                spaceBefore=10
            ))
        
        # Body text style
        if 'CustomBody' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBody',
                fontName=font_name,
                fontSize=12,
                leading=14,
                alignment=0  # Left align
            ))

    def add_heading(self, text, level=1):
        """Add heading"""
        style = 'CustomHeading1' if level == 1 else 'CustomHeading2'
        self.pdf_elements.append(Paragraph(text, self.styles[style]))
        self.pdf_elements.append(Spacer(1, 12))

    def add_paragraph(self, text):
        """Add paragraph"""
        self.pdf_elements.append(Paragraph(text, self.styles['CustomBody']))
        self.pdf_elements.append(Spacer(1, 12))

    def add_table(self, data, colWidths=None):
        """Add table"""
        # Use default font
        font_name = 'Helvetica'
            
        # Set table style
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        if not colWidths:
            colWidths = [self.doc.width/len(data[0])] * len(data[0])
            
        if len(data) > 25:
            # Keep header and randomly select 25 rows of data
            header = data[0]
            body = data[1:]
            selected_rows = random.sample(body, min(24, len(body)))
            
            # If first column is numeric, sort by first column
            try:
                # Try to convert first column to numbers
                first_col_nums = [float(row[0]) for row in selected_rows]
                # Sort by first column values
                sorted_indices = np.argsort(first_col_nums)
                selected_rows = [selected_rows[i] for i in sorted_indices]
            except (ValueError, TypeError):
                # If conversion fails, keep random order
                pass
                
            truncated_data = [header] + selected_rows
            # Add info row
            info_row = [f"(Showing 25 random samples, total {len(data)} records)" for _ in range(len(data[0]))]
            truncated_data.append(info_row)
            table = Table(truncated_data, colWidths=colWidths, style=style)
        else:
            table = Table(data, colWidths=colWidths, style=style)
        
        self.pdf_elements.append(table)
        self.pdf_elements.append(Spacer(1, 12))


    def figure_to_image(self, fig):
        """Convert matplotlib figure to reportlab image"""
        buf = io.BytesIO()
        
        # Get available space on PDF page
        available_width = self.doc.width * 0.9  # Leave 10% margin
        available_height = self.doc.height * 0.6  # Leave 40% for other content
        
        # Calculate current image size
        fig_size = fig.get_size_inches()
        dpi = 100  # Lower DPI to reduce file size
        
        # Calculate actual pixel size of image
        img_width = fig_size[0] * dpi
        img_height = fig_size[1] * dpi
        
        # Calculate scaling ratio
        width_ratio = available_width / img_width
        height_ratio = available_height / img_height
        scale = min(width_ratio, height_ratio, 1.0)  # Don't enlarge, only shrink
        
        # Save image
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   pad_inches=0.1)
    
        
        buf.seek(0)
        img = Image(buf)
        
        # Set image display size in PDF
        img.drawWidth = img_width * scale
        img.drawHeight = img_height * scale
        
        return img


    def analyze_and_generate_report(self):
        try:
            """Execute analysis and generate PDF report"""
            self.add_heading("Data Analysis Report", 1)
                
            # 1. Basic dataset information
            self.add_heading("1. Dataset Basic Information", 2)
            self._analyze_dataset_info()

            self.add_heading("Spectral Data", 2)
            self._plot_all_spectra()

            self.add_heading("Noise Levels", 2)
            self._analyze_noise_levels()

            self.add_heading("Data Relationship Analysis", 2)
            self._plot_data_relationships()

            self.add_heading("Pairwise Distribution", 2)
            self._plot_pairwise_relationships()
            
            # 2. Spectral data analysis
            self.add_heading("2. Spectral Data Analysis", 2)
            self._analyze_spectral_data()
            
            # 3. Other features analysis
            self.add_heading("3. Other Features Analysis", 2)
            try:
                self._analyze_other_features()
            except Exception as e:
                self.add_paragraph(f"No other feature data or other feature analysis failed: {e}")
                self.add_paragraph(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            
            # 4. Temporal pattern analysis
            if 'collection_date' in self.dataset:
                self.add_heading("4. Temporal Pattern Analysis", 2)
                daily_stats = self._analyze_temporal_patterns()
                self.add_paragraph("Temporal pattern analysis results:")
                
                # Add temporal pattern statistics table
                stats_table = [['Date', 'Mean Intensity', 'Standard Deviation', 'Sample Count']]
                for _, row in daily_stats.iterrows():
                    stats_table.append([
                        str(row['date']),  # Direct access to date column
                        f"{row[('mean_intensity', 'mean')]:.4f}",  # Correct access to multi-level index
                        f"{row[('mean_intensity', 'std')]:.4f}",   # Correct access to multi-level index
                        str(int(row[('mean_intensity', 'count')]))  # Correct access to multi-level index
                    ])
                self.add_table(stats_table)
            
            # 5. Volunteer pattern analysis
            if 'volunteer' in self.dataset:
                self.add_heading("5. Volunteer Pattern Analysis", 2)
                volunteer_stats = self._analyze_volunteer_patterns()
                self.add_paragraph("Volunteer pattern analysis results:")
                
                # Add volunteer statistics table
                stats_table = [['Volunteer ID', 'Mean Intensity', 'Standard Deviation', 'Sample Count']]
                for _, row in volunteer_stats.iterrows():
                    stats_table.append([
                        str(row['volunteer']),
                        f"{row['mean_intensity']['mean']:.4f}",
                        f"{row['mean_intensity']['std']:.4f}",
                        str(int(row['mean_intensity']['count']))
                    ])
                self.add_table(stats_table)
            
            # 6. Correlation analysis
            if len(self.dataset.keys()) > 1:
                self.add_heading("6. Feature Correlation Analysis", 2)
                try:
                    self._analyze_correlations()
                except Exception as e:
                    self.add_paragraph(f"Feature correlation analysis failed: {e}")
                    self.add_paragraph(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

            # 
            self._analyze_spectral_details()

            # 7. Model analysis
            self.add_heading("7. Model Analysis", 2)
            try:
                self._analyze_models()
            except Exception as e:
                print(f"{sys._getframe().f_lineno}: analyze models failed: {str(e)}")
                self.add_paragraph(f"Model analysis failed: {e}")
                self.add_paragraph(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

        
        
        # Generate PDF file
        # try:
            self.doc.build(self.pdf_elements)
            print(f"Report generated: {self.output_path}")
        except Exception as e:
            print(f"Error occurred at: {e.__traceback__.tb_frame.f_code.co_filename} line {e.__traceback__.tb_lineno}")
            raise
    
    def _plot_data_relationships(self):
        """Plot relationship graphs between any two data types"""
        # Get all plottable data columns
        plottable_data = {}
        # Add spectral intensity
        spectral_intensities = np.mean(self.dataset['spectra'], axis=1)
        plottable_data['spectral_intensity'] =  pd.to_numeric(spectral_intensities, errors='coerce')
        
        
        for key, value in self.dataset.items():
            if key != 'spectra':  # Exclude spectral data
                try:
                    # Try to convert to numeric type
                    numeric_data = pd.to_numeric(value, errors='coerce')
                    if  pd.api.types.is_numeric_dtype(numeric_data):  # If not all NA, treat as numeric
                        plottable_data[key] = numeric_data
                    else:  # If all conversion failed, treat as categorical data
                        plottable_data[key] = pd.Series(value).astype(str)
                except Exception as e:
                    print(f"Error occurred at: {e.__traceback__.tb_frame.f_code.co_filename} line {e.__traceback__.tb_lineno}")
                    # If conversion failed, treat as categorical data

                    plottable_data[key] = pd.Series(value).astype(str)

                    
        if len(plottable_data) < 2:
            self.add_paragraph("Less than 2 variables available for relationship analysis in the dataset, unable to perform relationship visualization.")
            return

        # Visualize all possible data pairs
        for i, (key1, data1) in enumerate(plottable_data.items()):
            for key2, data2 in list(plottable_data.items())[i+1:]:
                fig = plt.figure(figsize=(12, 6))
                
                # Choose appropriate visualization method based on data types
                if pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2):
                    # Numeric vs Numeric: scatter plot
                    plt.scatter(data1, data2, alpha=0.5)
                    
                    # Add trend line
                    try:
                        z = np.polyfit(data1, data2, 1)
                        p = np.poly1d(z)
                        plt.plot(data1, p(data1), "r--", alpha=0.8)
                        
                        # Calculate correlation coefficient
                        corr, p_val = scipy.stats.pearsonr(data1, data2)
                        plt.text(0.05, 0.95, 
                                f'Correlation: {corr:.3f}\np-value: {p_val:.3e}',
                                transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.8))
                    except:
                        pass

                elif pd.api.types.is_numeric_dtype(data1) and not pd.api.types.is_numeric_dtype(data2):
                    # Numeric vs Categorical: box plot
                    df_temp = pd.DataFrame({'value': data1, 'category': data2})
                    unique_categories = df_temp['category'].unique()
                    data_by_category = [df_temp[df_temp['category'] == cat]['value'].dropna().values 
                                      for cat in unique_categories]
                    # 过滤掉空的类别
                    valid_categories = []
                    valid_data = []
                    for cat, data in zip(unique_categories, data_by_category):
                        if len(data) > 0:
                            valid_categories.append(str(cat))
                            valid_data.append(data)
                    
                    if valid_data:
                        plt.boxplot(valid_data, labels=valid_categories)
                        plt.xticks(rotation=45)
                        
                        # 进行方差分析
                        if len(valid_data) >= 2:  # At least two groups needed for ANOVA
                            try:
                                # Ensure each group has at least two valid values
                                valid_groups = [group for group in valid_data if len(group) >= 2]
                                if len(valid_groups) >= 2:
                                    f_stat, p_val = scipy.stats.f_oneway(*valid_groups)
                                    plt.text(0.05, 0.95, 
                                            f'ANOVA Test:\nF-statistic: {f_stat:.3f}\np-value: {p_val:.3e}',
                                            transform=plt.gca().transAxes,
                                            bbox=dict(facecolor='white', alpha=0.8))
                                else:
                                    plt.text(0.05, 0.95, 
                                            'Cannot perform ANOVA test:\nEach group needs at least 2 samples',
                                            transform=plt.gca().transAxes,
                                            bbox=dict(facecolor='white', alpha=0.8))
                            except Exception as e:
                                plt.text(0.05, 0.95, 
                                        f'ANOVA test failed:\n{str(e)}',
                                        transform=plt.gca().transAxes,
                                        bbox=dict(facecolor='white', alpha=0.8))
                    else:
                        plt.text(0.5, 0.5, 'Not enough valid data for analysis',
                                horizontalalignment='center',
                                verticalalignment='center')

                elif pd.api.types.is_numeric_dtype(data2) and not pd.api.types.is_numeric_dtype(data1):
                    # Categorical vs Numeric: box plot
                    df_temp = pd.DataFrame({'value': data2, 'category': data1})
                    unique_categories = df_temp['category'].unique()
                    data_by_category = [df_temp[df_temp['category'] == cat]['value'].dropna().values 
                                      for cat in unique_categories]
                    
                    # 过滤掉空的类别
                    valid_categories = []
                    valid_data = []
                    for cat, data in zip(unique_categories, data_by_category):
                        if len(data) > 0:
                            valid_categories.append(str(cat))
                            valid_data.append(data)
                    
                    if valid_data:
                        plt.boxplot(valid_data, labels=valid_categories)
                        plt.xticks(rotation=45)
                        
                        # 进行方差分析
                        if len(valid_data) >= 2:  # At least two groups needed for ANOVA
                            try:
                                f_stat, p_val = scipy.stats.f_oneway(*valid_data)
                                plt.text(0.05, 0.95, 
                                        f'ANOVA Test:\nF-statistic: {f_stat:.3f}\np-value: {p_val:.3e}',
                                        transform=plt.gca().transAxes,
                                        bbox=dict(facecolor='white', alpha=0.8))
                            except Exception as e:
                                print(f"ANOVA analysis failed: {str(e)}")
                    else:
                        plt.text(0.5, 0.5, 'No valid data for analysis',
                                horizontalalignment='center',
                                verticalalignment='center')

                else:
                    # Categorical vs Categorical: heatmap
                    try:
                        # Ensure data are string type
                        df_temp = pd.DataFrame({
                            'var1': pd.Series(data1).astype(str),
                            'var2': pd.Series(data2).astype(str)
                        })
                        
                        # Create contingency table
                        contingency_table = pd.crosstab(df_temp['var1'], df_temp['var2'])
                        
                        # Create figure with subplots
                        fig = plt.figure(figsize=(15, 10))
                        gs = plt.GridSpec(2, 2, width_ratios=[0.2, 0.8], height_ratios=[0.8, 0.2])
                        
                        # Main heatmap
                        ax_main = plt.subplot(gs[0, 1])
                        if contingency_table.shape[0] * contingency_table.shape[1] > 100:
                            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd',
                                      annot_kws={'size': 8})
                        else:
                            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        
                        # Left bar chart (var1 distribution)
                        ax_left = plt.subplot(gs[0, 0])
                        var1_counts = df_temp['var1'].value_counts()
                        ax_left.barh(range(len(var1_counts)), var1_counts.values)
                        ax_left.set_yticks([])
                        ax_left.invert_xaxis()
                        
                        # Bottom bar chart (var2 distribution)
                        ax_bottom = plt.subplot(gs[1, 1])
                        var2_counts = df_temp['var2'].value_counts()
                        ax_bottom.bar(range(len(var2_counts)), var2_counts.values)
                        ax_bottom.set_xticks([])
                        
                        # Perform chi-square test
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                            plt.text(1.05, 0.95,
                                   f'Chi-square test:\nStatistic: {chi2:.3f}\np-value: {p_val:.3e}',
                                   transform=ax_main.transAxes,
                                   bbox=dict(facecolor='white', alpha=0.8))
                    except Exception as e:
                        plt.text(0.5, 0.5, f'Cannot create heatmap: {str(e)}',
                                horizontalalignment='center',
                                verticalalignment='center')
                    
                    # 进行卡方检验
                    try:
                        chi2, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                        plt.text(1.05, 0.95, 
                                                                    f'Chi-square test:\nStatistic: {chi2:.3f}\np-value: {p_val:.3e}',
                                transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.8))
                    except:
                        pass

                # plt.title(f'{key1} vs {key2} relationship')
                plt.xlabel(key1)
                plt.ylabel(key2)
                plt.grid(True, alpha=0.3)
                
                # Adjust layout to avoid label overlap
                plt.tight_layout()
                # Add to PDF
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # Add statistical description
                self.add_paragraph(f"\nRelationship analysis between {key1} and {key2}:")
                
                # Add different statistical descriptions based on data type
                if data1.dtype.kind in 'iufc' and data2.dtype.kind in 'iufc':
                    # Add statistical description for numeric variables
                    stats_table = [['Statistic', 'Value']]
                    stats_table.append(['Sample Count', str(len(data1))])
                    
                    if corr is not None:
                        stats_table.append(['Pearson Correlation', f"{corr:.4f}"])
                        stats_table.append(['Correlation p-value', f"{p_val:.4e}"])
                    
                    self.add_table(stats_table)
                    
                elif data1.dtype.kind not in 'iufc' or data2.dtype.kind not in 'iufc':
                    # Add statistical description for categorical variables
                    if 'f_stat' in locals():
                        stats_table = [['Statistic', 'Value']]
                        stats_table.append(['ANOVA F-statistic', f"{f_stat:.4f}"])
                        stats_table.append(['ANOVA p-value', f"{p_val:.4e}"])
                        self.add_table(stats_table)
                    
                    # Add basic descriptive statistics
                    if data1.dtype.kind in 'iufc':
                        numeric_data = data1
                        category_data = data2
                    else:
                        numeric_data = data2
                        category_data = data1
                    
                    # Calculate descriptive statistics for each category
                    desc_table = [['Category', 'Sample Count', 'Mean', 'Std Dev', 'Min Value', 'Max Value']]
                    
                    # Ensure data is numeric type
                    numeric_data = pd.to_numeric(numeric_data, errors='coerce')
                    
                    # Use pandas for group statistics, avoiding null and non-numeric issues
                    df = pd.DataFrame({'numeric': numeric_data, 'category': category_data})
                    for cat in df['category'].unique():
                        cat_data = df[df['category'] == cat]['numeric'].dropna()
                        if len(cat_data) > 0:
                            desc_table.append([
                                str(cat),
                                str(len(cat_data)),
                                f"{cat_data.mean():.4f}",
                                f"{cat_data.std():.4f}",
                                f"{cat_data.min():.4f}", 
                                f"{cat_data.max():.4f}"
                            ])
                        else:
                            desc_table.append([
                                str(cat),
                                '0',
                                'N/A',
                                'N/A', 
                                'N/A',
                                'N/A'
                            ])
                    self.add_table(desc_table)
    
    def _plot_all_spectra(self):
        """Plot spectral data overlay grouped by different labels"""
        # Get all non-spectral data columns as labels
        label_columns = [key for key in self.dataset.keys() if key != 'spectra']
        
        for label_column in label_columns:
            try:
                # Create figure
                
                # Get unique label values
                unique_labels = np.unique(self.dataset[label_column])
                
                # Set different colors for different labels
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                fig = plt.figure(figsize=(15, 20))
                gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
                
                # Upper subplot: spectra grouped by labels
                ax1 = plt.subplot(gs[0])
                
                # Draw spectra for each label
                for label, color in zip(unique_labels, colors):
                    # Get spectral data indices for this label
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    # Calculate mean spectrum and standard deviation for this label
                    label_mean = np.mean(label_spectra, axis=0)
                    label_std = np.std(label_spectra, axis=0)
                    
                    # Draw all spectra for this label (high transparency)
                    alpha_value = max(0.05, 1.0 / np.sqrt(len(label_spectra)))
                    for spectrum in label_spectra:
                        ax1.plot(spectrum, '-', color=color, alpha=alpha_value, linewidth=0.5)
                    
                    # # Draw mean spectrum for this label (opaque)
                    # ax1.plot(label_mean, '-', color=color, linewidth=2, 
                    #         label=f'{label_column}={label} (n={len(label_spectra)})')
                    
                    # # Draw standard deviation range
                    # ax1.fill_between(range(len(label_mean)),
                    #             label_mean - label_std,
                    #             label_mean + label_std,
                    #             color=color, alpha=0.2)
                
                # Set labels and title for upper subplot
                ax1.set_title(f'Spectral Data Grouped by {label_column}')
                ax1.set_xlabel('Wavelength Index')
                ax1.set_ylabel('Spectral Intensity')
                ax1.grid(True, alpha=0.3)
                # Only add legend if there are labeled artists
                handles, labels = ax1.get_legend_handles_labels()
                if handles:
                    ax1.legend()
                
                # Middle subplot: mean values for each group
                ax2 = plt.subplot(gs[1])
                
                # Draw mean values for each label
                for label, color in zip(unique_labels, colors):
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    # Calculate mean
                    label_mean = np.mean(label_spectra, axis=0)
                    
                    # Draw mean
                    ax2.plot(label_mean, '-', color=color, label=f'{label_column}={label}')
                
                # Set labels for middle subplot
                ax2.set_xlabel('Wavelength Index')
                ax2.set_ylabel('Mean Value')
                ax2.grid(True, alpha=0.3)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Third subplot: normalized mean values
                ax4 = plt.subplot(gs[2])
                
                # Collect mean data for all labels
                all_means = []
                for label in unique_labels:
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    label_mean = np.mean(label_spectra, axis=0)
                    all_means.append(label_mean)
                
                # Convert all mean data to numpy array
                all_means = np.array(all_means)
                
                # Normalize each feature (wavelength point) separately
                normalized_means = np.zeros_like(all_means)
                for feature_idx in range(all_means.shape[1]):  # Iterate through each feature (wavelength point)
                    feature_values = all_means[:, feature_idx]  # Get values for all labels at this feature
                    max_val = np.max(feature_values)  # Maximum value for this feature
                    min_val = np.min(feature_values)  # Minimum value for this feature
                    if max_val != min_val:  # Avoid division by zero
                        normalized_means[:, feature_idx] = (feature_values - min_val) / (max_val - min_val)
                    else:
                        normalized_means[:, feature_idx] = feature_values
                
                # Draw normalized mean values for each label
                for label, color, norm_mean in zip(unique_labels, colors, normalized_means):
                    ax4.plot(norm_mean, '-', color=color, label=f'{label_column}={label}')
                
                # Set labels for third subplot
                ax4.set_xlabel('Wavelength Index (Feature)')
                ax4.set_ylabel('Normalized Mean (0-1)')
                ax4.grid(True, alpha=0.3)
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Bottom subplot: coefficient of variation for each group
                ax3 = plt.subplot(gs[3])
                
                # Calculate and draw coefficient of variation for each label
                for label, color in zip(unique_labels, colors):
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    # Calculate coefficient of variation
                    label_mean = np.mean(label_spectra, axis=0)
                    label_std = np.std(label_spectra, axis=0)
                    cv = label_std / np.abs(label_mean) * 100
                    
                    # Draw coefficient of variation
                    ax3.plot(cv, '-', color=color, label=f'{label_column}={label}')
                
                # Set labels for bottom subplot
                ax3.set_xlabel('Wavelength Index')
                ax3.set_ylabel('Coefficient of Variation (%)')
                ax3.grid(True, alpha=0.3)
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                # Add statistical information for each label
                stats_table = [['Label Value', 'Sample Count', 'Mean Intensity', 'Standard Deviation', 'Mean CV (%)']]
                for label in unique_labels:
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    mean_intensity = np.mean(label_spectra)
                    std_intensity = np.std(label_spectra)
                    mean_cv = np.mean(np.std(label_spectra, axis=0) / 
                                    np.abs(np.mean(label_spectra, axis=0))) * 100
                    
                    stats_table.append([
                        str(label),
                        str(len(label_spectra)),
                        f"{mean_intensity:.4f}",
                        f"{std_intensity:.4f}",
                        f"{mean_cv:.2f}"
                    ])
                
                # plt.tight_layout()
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # Add statistics table for this label
                self.add_paragraph(f"\n{label_column} Grouping Statistics:")
                self.add_table(stats_table)
            except Exception as e:
                print(f"{sys._getframe().f_lineno}: draw spectra failed: {str(e)}")

    def _analyze_models(self):
        """Analyze predictive modeling for different data types, supporting group analysis based on categorical variables"""
        # Prepare data
        chemical_features = {}
        categorical_features = {}
        
        for key, value in self.dataset.items():
            try:
                if key != 'spectra':
                    if pd.api.types.is_numeric_dtype(value):
                        chemical_features[key] = value
                    else:
                        try:
                            # Try to convert to numeric type
                            numeric_value = pd.to_numeric(value)
                            chemical_features[key] = numeric_value
                        except:
                            categorical_features[key] = value
            except Exception as e:
                print(f"{sys._getframe().f_lineno}: analyze models failed: {str(e)}")
        
        if not chemical_features:
            return
        
        spectral_data = self.spectral_data
        # 对每个数值特征进行预测分析
        for target_name, target_values in chemical_features.items():
            self.add_heading(f"Model Prediction Analysis for {target_name}", 3)
            
            # 1. 使用光谱数据进行常规预测
            self.add_heading("Prediction Based on Spectral Data", 4)
            spectral_prediction = self._analyze_with_spectral(
                spectral_data, 
                target_values,
                target_name
            )
            
            # 2. 对每个分类变量进行分组预测分析
            if categorical_features:
                for cat_name, cat_values in categorical_features.items():
                    try:
                        self.add_heading(f"Prediction Analysis for {target_name} Grouped by {cat_name}", 4)
                        self._analyze_by_group(
                            spectral_data,
                            target_values,
                            cat_values,
                            target_name,
                            cat_name
                        )
                    except Exception as e:
                        print(f"{sys._getframe().f_lineno}: analyze by group failed: {str(e)}")
                    

            # 3. 如果有日期数据，进行时间序列分析
            if 'collection_date' in self.dataset:
                self.add_heading(f"Time Series Prediction Analysis for {target_name}", 4)
                time_values = pd.to_datetime(self.dataset['collection_date'])
                try:
                    self._analyze_by_time(
                        spectral_data,
                        target_values,
                        time_values,
                        target_name,
                        'collection_date'
                    )
                except Exception as e:
                    print(f"{sys._getframe().f_lineno}: analyze by time failed: {str(e)}")

    def _categorize_features(self):
        """将数据集特征分类为不同类型"""
        feature_types = {
            'numeric': {},    # 连续数值型特征
            'categorical': {}, # 离散分类型特征
            'temporal': {},   # 时间型特征
            'spectral': {}    # 光谱数据
        }
        
        for key, value in self.dataset.items():
            if key == 'spectra':
                feature_types['spectral'][key] = {
                    'data': value,
                    'shape': value.shape
                }
                continue
            
            # Try to convert to date type
            try:
                pd.to_datetime(value)
                feature_types['temporal'][key] = {
                    'data': pd.to_datetime(value),
                    'unique_count': len(pd.unique(value))
                }
                continue
            except:
                pass
            
            # Check if it's numeric type
            if pd.api.types.is_numeric_dtype(value):
                feature_types['numeric'][key] = {
                    'data': value,
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'unique_count': len(pd.unique(value))
                }
            else:
                # Non-numeric types are treated as categorical variables
                feature_types['categorical'][key] = {
                    'data': value,
                    'unique_count': len(pd.unique(value)),
                    'categories': pd.unique(value)
                }
        
        return feature_types

    def _analyze_with_spectral(self, spectral_data, target_values, target_name):
        """Perform prediction analysis using spectral data"""
        # 定义模型
        models = {
            'PLS Regression': PLSRegression(n_components=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression()
        }
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            spectral_data, target_values, test_size=0.2, random_state=42
        )
        
        results = {}
        fig = plt.figure(figsize=(16, 12))
        
        for i, (name, model) in enumerate(models.items(), 1):
            # 训练和预测
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            
            # 绘制预测散点图
            plt.subplot(2, 2, i)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{name} Prediction Results')
            
            plt.text(0.05, 0.95, 
                    f'R2 = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 添加结果表格
        results_table = [['Model', 'R2 Score', 'RMSE', 'MAE']]
        for name, metrics in results.items():
            results_table.append([
                name,
                f"{metrics['R2']:.3f}",
                f"{metrics['RMSE']:.3f}",
                f"{metrics['MAE']:.3f}"
            ])
        
        self.add_table(results_table)
        return results

    def _analyze_by_group(self, spectral_data, target_values, group_values, 
                        target_name, group_name):
        """按分组进行预测分析"""
        self.add_heading(f"Prediction Analysis for {target_name} Grouped by {group_name}", 4)
        
        unique_groups = np.unique(group_values)
        group_results = {}
        
        # 创建分组结果图
        n_groups = len(unique_groups)
        n_cols = min(2, n_groups)
        n_rows = (n_groups + 1) // 2
        fig = plt.figure(figsize=(15 * n_cols, 10 * n_rows))
        
        for idx, group in enumerate(unique_groups, 1):
            # 获取该组的数据
            mask = group_values == group
            group_spectral = spectral_data[mask]
            group_target = target_values[mask]
            
            if len(group_target) < 10:  # 样本太少的组跳过
                continue
            
            # 为该组训练模型
            X_train, X_test, y_train, y_test = train_test_split(
                group_spectral, group_target, test_size=0.2, random_state=42
            )
            
            # 使用PLS回归作为示例模型
            model = PLSRegression(n_components=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算性能指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            group_results[group] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            
            # 绘制该组的预测结果
            plt.subplot(n_rows, n_cols, idx)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Prediction Results for {group_name}={group}')
            
            plt.text(0.05, 0.95, 
                    f'Sample Count: {len(group_target)}\n'
                    f'R2 = {r2:.3f}\n'
                    f'RMSE = {rmse:.3f}\n'
                    f'MAE = {mae:.3f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 添加分组比较表格
        comparison_table = [[f'{group_name}', 'sample_num', 'R2', 'RMSE', 'MAE']]
        for group, metrics in group_results.items():
            comparison_table.append([
                str(group),
                str(len(spectral_data[group_values == group])),
                f"{metrics['R2']:.3f}",
                f"{metrics['RMSE']:.3f}",
                f"{metrics['MAE']:.3f}"
            ])
        
        self.add_table(comparison_table)

    def _analyze_by_time(self, spectral_data, target_values, time_values, 
                        target_name, time_name):
        """按时间进行预测分析"""
        self.add_heading(f"Time Series Prediction Analysis for {target_name} Based on {time_name}", 4)
        
        # 将数据按时间排序
        sorted_indices = np.argsort(time_values)
        sorted_spectral = spectral_data[sorted_indices]
        sorted_target = target_values[sorted_indices]
        sorted_time = time_values[sorted_indices]
        
        # 按时间划分训练集和测试集（使用最后20%的数据作为测试集）
        split_idx = int(len(sorted_target) * 0.8)
        X_train = sorted_spectral[:split_idx]
        X_test = sorted_spectral[split_idx:]
        y_train = sorted_target[:split_idx]
        y_test = sorted_target[split_idx:]
        time_test = sorted_time[split_idx:]
        
        # 训练模型
        model = PLSRegression(n_components=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # 绘制时间序列预测结果
        fig = plt.figure(figsize=(15, 8))
        plt.scatter(time_test, y_test, label='Actual Values', alpha=0.5)
        plt.scatter(time_test, y_pred, label='Predicted Values', alpha=0.5)
        plt.xlabel(time_name)
        plt.ylabel(target_name)
        plt.title(f'Time Series Prediction for {target_name}')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.text(0.05, 0.95, 
                f'R2 = {r2:.3f}\n'
                f'RMSE = {rmse:.3f}\n'
                f'MAE = {mae:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)

    def _plot_pairwise_relationships(self):
        """绘制数据集中变量的分组分布图"""
        # 获取所有数值型变量和分类变量
        numeric_data = {}
        categorical_data = {}
        
        # 计算光谱强度(每个样本的均值)
        spectral_intensities = np.mean(self.dataset['spectra'], axis=1)
        numeric_data['spectral_intensity'] = spectral_intensities
        
        for key, value in self.dataset.items():
            if key != 'spectra':
                if pd.api.types.is_numeric_dtype(value):
                    numeric_data[key] = value
                else:
                    try:
                        pd.to_datetime(value)  # 尝试转换为日期
                        categorical_data[key] = value
                    except:
                        categorical_data[key] = value
        
        if len(numeric_data) < 1 or len(categorical_data) < 1:
            self.add_paragraph("The dataset lacks sufficient numeric or categorical variables to plot group distribution charts.")
            return
            
        # 创建数据框
        df = pd.DataFrame({**numeric_data, **categorical_data})
        
        # 为每个数值变量和分类变量的组合创建分布图
        for num_col in numeric_data.keys():
            for cat_col in categorical_data.keys():
                fig = plt.figure(figsize=(15, 6))
                
                # 创建子图
                plt.subplot(1, 2, 1)
                # 按组绘制核密度估计图
                for group in df[cat_col].unique():
                    group_data = df[df[cat_col] == group][num_col]
                    sns.kdeplot(data=group_data, label=str(group))
                plt.title(f'Density Distribution of {num_col} under Different {cat_col}')
                plt.xlabel(num_col)
                plt.ylabel('Density')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                # 按组绘制小提琴图
                sns.violinplot(x=cat_col, y=num_col, data=df)
                plt.title(f'Distribution of {num_col} under Different {cat_col}')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                # Add to PDF
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # 添加分组描述性统计
                self.add_paragraph(f"\nStatistical description of {num_col} grouped by {cat_col}:")
                stats_table = [['Group', 'Sample Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']]
                
                for group in df[cat_col].unique():
                    group_data = df[df[cat_col] == group][num_col]
                    if len(group_data) > 0:
                        stats = group_data.describe()
                        stats_table.append([
                            str(group),
                            f"{stats['count']:.0f}",
                            f"{stats['mean']:.3f}",
                            f"{stats['std']:.3f}",
                            f"{stats['min']:.3f}",
                            f"{stats['25%']:.3f}",
                            f"{stats['50%']:.3f}",
                            f"{stats['75%']:.3f}",
                            f"{stats['max']:.3f}"
                        ])
                    
                self.add_table(stats_table)

    def _analyze_dataset_info(self):
        """分析数据集基本信息"""
        # 创建数据集信息表格
        dataset_info = [
            ['feature_name', 'data_type', 'shape', 'non_null_count'],
        ]
        
        for key, value in self.dataset.items():
            dataset_info.append([
                key,
                str(value.dtype),
                str(value.shape),
                str(np.sum(~pd.isna(value)))
            ])
        
        self.add_paragraph("The dataset contains the following features:")
        self.add_table(dataset_info)

    def _analyze_spectral_data(self):
        """Analyze spectral data"""
        # 基本统计信息
        stats = {
            'sample_num': self.n_samples,
            'feature_num': self.n_features,
            'spectral_data_stats': {
                'mean': f"{np.mean(self.spectral_data):.4f}",
                'std': f"{np.std(self.spectral_data):.4f}",
                'min': f"{np.min(self.spectral_data):.4f}",
                'max': f"{np.max(self.spectral_data):.4f}"
            }
        }
        
        for key, value in stats.items():
            if isinstance(value, dict):
                self.add_paragraph(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.add_paragraph(f"    {sub_key}: {sub_value}")
            else:
                self.add_paragraph(f"{key}: {value}")
        
        # 平均光谱图
        fig = plt.figure(figsize=(12, 6))
        mean_spectrum = np.mean(self.spectral_data, axis=0)
        std_spectrum = np.std(self.spectral_data, axis=0)
        
        plt.plot(mean_spectrum, 'b-', label='mean_spectrum')
        plt.fill_between(range(len(mean_spectrum)),
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.2,
                        color='b',
                                                 label='±1 Std Dev')
        
        plt.xlabel('wavelength_index')
        plt.ylabel('spectral_intensity')
        plt.title('mean_spectrum_and_its_variation_range')
        plt.legend()
        plt.grid(True)
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # PCA分析
        self.add_heading("Principal Component Analysis (PCA)", 3)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.spectral_data)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        # 绘制解释方差比
        fig = plt.figure(figsize=(10, 5))
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 
                'bo-')
        plt.xlabel('principal_component_num')
        plt.ylabel('cumulative_explained_variance_ratio')
        plt.title('PCA_cumulative_explained_variance_ratio')
        plt.grid(True)
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        self.add_paragraph(f"the_first_three_principal_components_explained_variance_ratio：")
        for i, ratio in enumerate(explained_variance_ratio[:3], 1):
            self.add_paragraph(f"PC{i}: {ratio:.4f}")

    def _analyze_other_features(self):
        """分析其他特征"""
        for key, value in self.dataset.items():
            if key != 'spectra':
                self.add_heading(f"{key} Feature Analysis", 3)
                
                if value.dtype.kind in 'iufc':  # 数值型数据
                    # 统计信息
                    stats = {
                        'mean': np.mean(value),
                        'std': np.std(value),
                        'min': np.min(value),
                        'max': np.max(value),
                        'median': np.median(value)
                    }
                    
                    stats_table = [['statistic', 'value']]
                    for stat_name, stat_value in stats.items():
                        stats_table.append([stat_name, f"{stat_value:.4f}"])
                    
                    self.add_table(stats_table)
                    
                    # 绘制分布图
                    fig = plt.figure(figsize=(10, 5))
                    plt.hist(value, bins=30, edgecolor='black')
                    plt.title(f"{key}distribution_histogram")
                    plt.xlabel(key)
                    plt.ylabel('frequency')
                    plt.grid(True)
                    self.pdf_elements.append(self.figure_to_image(fig))
                    plt.close(fig)
                
                else:  # 类别型数据
                    # 统计每个类别的数量
                    value_counts = pd.Series(value).value_counts()
                    
                    counts_table = [['category', 'count']]
                    for cat, count in value_counts.items():
                        counts_table.append([str(cat), str(count)])
                    
                    self.add_table(counts_table)
                    
                    # 绘制条形图
                    fig = plt.figure(figsize=(10, 5))
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                    plt.title(f"{key}category_distribution")
                    plt.xlabel(key)
                    plt.ylabel('count')
                    plt.grid(True)
                    plt.tight_layout()
                    self.pdf_elements.append(self.figure_to_image(fig))
                    plt.close(fig)
    def _analyze_spectral_details(self):
        """详细分析光谱数据特征"""
        # 1. 计算并绘制一阶导数和二阶导数
        spectra = self.spectral_data
        # 计算导数
        first_derivative = np.gradient(spectra, axis=1)
        second_derivative = np.gradient(first_derivative, axis=1)
        
        # 绘制导数图
        fig = plt.figure(figsize=(12, 8))
        
        # 原始光谱
        plt.subplot(3, 1, 1)
        mean_spectrum = np.mean(spectra, axis=0)
        std_spectrum = np.std(spectra, axis=0)
        plt.plot(mean_spectrum, 'b-', label='mean_spectrum')
        plt.fill_between(range(len(mean_spectrum)),
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.2,
                        color='b',
                                                 label='±1 Std Dev')
        plt.title('original_spectrum')
        plt.xlabel('wavelength_index')
        plt.ylabel('spectral_intensity')
        plt.grid(True)
        plt.legend()
        
        # 一阶导数
        plt.subplot(3, 1, 2)
        mean_first_deriv = np.mean(first_derivative, axis=0)
        std_first_deriv = np.std(first_derivative, axis=0)
        plt.plot(mean_first_deriv, 'r-', label='mean_first_derivative')
        plt.fill_between(range(len(mean_first_deriv)),
                        mean_first_deriv - std_first_deriv,
                        mean_first_deriv + std_first_deriv,
                        alpha=0.2,
                        color='r',
                                                 label='±1 Std Dev')
        plt.title('first_derivative')
        plt.xlabel('wavelength_index')
        plt.ylabel('first_derivative_value')
        plt.grid(True)
        plt.legend()
        
        # 二阶导数
        plt.subplot(3, 1, 3)
        mean_second_deriv = np.mean(second_derivative, axis=0)
        std_second_deriv = np.std(second_derivative, axis=0)
        plt.plot(mean_second_deriv, 'g-', label='mean_second_derivative')
        plt.fill_between(range(len(mean_second_deriv)),
                        mean_second_deriv - std_second_deriv,
                        mean_second_deriv + std_second_deriv,
                        alpha=0.2,
                        color='g',
                                                 label='±1 Std Dev')
        plt.title('second_derivative')
        plt.xlabel('wavelength_index')
        plt.ylabel('second_derivative_value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 2. 绘制光谱特征分布图
        fig = plt.figure(figsize=(12, 6))
        
        # 计算每个样本的统计特征
        mean_intensities = np.mean(spectra, axis=1)
        max_intensities = np.max(spectra, axis=1)
        min_intensities = np.min(spectra, axis=1)
        range_intensities = max_intensities - min_intensities
        
        # 创建箱线图
        data = [mean_intensities, max_intensities, min_intensities, range_intensities]
        labels = ['mean_intensity', 'max_intensity', 'min_intensity', 'intensity_range']
        
        plt.boxplot(data, labels=labels)
        plt.title('spectral_feature_distribution')
        plt.ylabel('intensity_value')
        plt.grid(True)
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 3. 添加统计信息到报告
        stats_table = [
            ['statistic', 'mean', 'std', 'min', 'max', 'median']
        ]
        
        features = {
            'original_spectrum': spectra.mean(axis=1),
            'first_derivative': first_derivative.mean(axis=1),
            'second_derivative': second_derivative.mean(axis=1)
        }
        
        for name, values in features.items():
            stats = [
                name,
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}",
                f"{np.median(values):.4f}"
            ]
            stats_table.append(stats)
        
        self.add_table(stats_table)
        
        # 4. 特征峰识别和标注
        peak_indices = scipy.signal.find_peaks(mean_spectrum)[0]
        valley_indices = scipy.signal.find_peaks(-mean_spectrum)[0]
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(mean_spectrum, 'b-', label='mean_spectrum')
        plt.plot(peak_indices, mean_spectrum[peak_indices], 'ro', label='peak')
        plt.plot(valley_indices, mean_spectrum[valley_indices], 'go', label='valley')
        
        # 标注主要峰值
        for idx in peak_indices:
            plt.annotate(f'Peak: {mean_spectrum[idx]:.2f}',
                        (idx, mean_spectrum[idx]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8)
        
        plt.title('spectral_feature_peak_identification')
        plt.xlabel('wavelength_index')
        plt.ylabel('spectral_intensity')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 记录峰值信息
        self.add_paragraph("main_peak_position：")
        peak_info = [['peak_type', 'wavelength_index', 'intensity']]
        
        for idx in peak_indices:
            peak_info.append(['peak', str(idx), f"{mean_spectrum[idx]:.4f}"])
        for idx in valley_indices:
            peak_info.append(['valley', str(idx), f"{mean_spectrum[idx]:.4f}"])
        
        self.add_table(peak_info)
    def _analyze_correlations(self):
        """分析光谱特征与理化值之间的相关性"""
        # 准备数据
        chemical_features = {}
        for key, value in self.dataset.items():
            if key != '光谱' and value.dtype.kind in 'iufc':
                chemical_features[key] = value
        
        if chemical_features:
            # 创建相关性分析结果
            correlations = {}
            p_values = {}
            
            # 对每个理化指标进行分析
            for chem_name, chem_value in chemical_features.items():
                # 计算每个波长点与该理化值的相关系数
                wave_correlations = []
                wave_p_values = []
                
                for i in range(self.spectral_data.shape[1]):
                    # 使用scipy.stats计算相关系数和p值
                    corr, p_val = scipy.stats.pearsonr(
                        self.spectral_data[:, i],
                        chem_value
                    )
                    wave_correlations.append(corr)
                    wave_p_values.append(p_val)
                
                correlations[chem_name] = wave_correlations
                p_values[chem_name] = wave_p_values
            
            # 绘制相关性图
            n_chemicals = len(chemical_features)
            fig = plt.figure(figsize=(12, 4 * n_chemicals))
            
            for idx, (chem_name, correlation) in enumerate(correlations.items(), 1):
                plt.subplot(n_chemicals, 1, idx)
                
                # 绘制相关系数曲线
                plt.plot(correlation, 'b-', label='correlation_coefficient')
                
                # 标记显著性区域
                significant = np.array(p_values[chem_name]) < 0.05
                if np.any(significant):
                    plt.fill_between(
                        range(len(correlation)),
                        np.where(significant, correlation, np.nan),
                        alpha=0.3,
                        color='r',
                        label='p < 0.05'
                    )
                
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.axhline(y=0.5, color='g', linestyle=':', alpha=0.5)
                plt.axhline(y=-0.5, color='g', linestyle=':', alpha=0.5)
                
                plt.title(f'correlation_analysis_between_spectral_and_{chem_name}')
                plt.xlabel('wavelength_index')
                plt.ylabel('correlation_coefficient')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)
            
            # 添加文字说明
            for chem_name, correlation in correlations.items():
                # 找出最强相关的波长点
                max_corr_idx = np.argmax(np.abs(correlation))
                max_corr = correlation[max_corr_idx]
                max_corr_p = p_values[chem_name][max_corr_idx]
                
                self.add_paragraph(f"correlation_analysis_result_of_{chem_name}：")
                self.add_paragraph(
                    f"the_most_strong_correlated_wavelength_index：{max_corr_idx}，"
                    f"correlation_coefficient：{max_corr:.4f}，"
                    f"p值：{max_corr_p:.4e}"
                )
                
                # 统计显著相关的波长数量
                sig_count = np.sum(np.array(p_values[chem_name]) < 0.05)
                self.add_paragraph(
                    f"the_number_of_wavelength_points_with_significant_correlation(p<0.05)：{sig_count}，"
                    f"the_ratio_of_significant_correlation_wavelength_points_to_total_wavelength_points：{sig_count/len(correlation)*100:.2f}%"
                )

    def _analyze_temporal_patterns(self):
        """analyze_temporal_patterns"""
        if 'collection_date' not in self.dataset:
            raise ValueError("the_dataset_is_missing_the_information_of_collection_date")
            
        # 将日期转换为datetime对象
        dates = pd.to_datetime(self.dataset['collection_date'])
        self.dates = dates
        # 计算每日平均光谱  
        daily_means = pd.DataFrame({
            'date': dates,
            'mean_intensity': np.mean(self.dataset['spectra'], axis=1)
        })
        # 按日期分组并计算统计量
        daily_stats = daily_means.groupby('date').agg({
            'mean_intensity': ['mean', 'std', 'count']
        }).reset_index()
        
        # 绘制时间序列图(带误差线)
        fig = plt.figure(figsize=(15, 6))
        plt.errorbar(daily_stats['date'],
                    daily_stats['mean_intensity']['mean'],
                    yerr=daily_stats['mean_intensity']['std'],
                    fmt='o-',
                    capsize=5)
        plt.xlabel('date')
        plt.ylabel('mean_spectral_intensity')
        plt.title('mean_spectral_intensity_over_time(daily_statistics)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 绘制折线图(每个样本)
        fig = plt.figure(figsize=(15, 6))
        plt.plot(daily_means['date'], daily_means['mean_intensity'], 
                'o-', alpha=0.5, markersize=5)
        plt.xlabel('date')
        plt.ylabel('mean_spectral_intensity')
        plt.title('mean_spectral_intensity_over_time(each_sample)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)

        # 绘制按顺序的光谱强度图
        fig = plt.figure(figsize=(15, 6))
        plt.plot(daily_means['mean_intensity'], 'b-')
        # 在每个日期变化点添加竖线
        date_changes = np.where(daily_means['date'].diff() != pd.Timedelta(0))[0]
        for idx in date_changes:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('sample_index')
        plt.ylabel('mean_spectral_intensity')
        plt.title('mean_spectral_intensity_over_time(by_collection_order)')
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)

        # 如果数据集中包含实测值,绘制实测值随时间变化的图
        if 'measured_value' in self.dataset:
            # 创建实测值时间序列数据
            measured_data = pd.DataFrame({
                'date': pd.to_datetime(self.dataset['collection_date']),
                'measured_value': self.dataset['measured_value']
            })
            
            # 计算每日实测值统计
            daily_measured = measured_data.groupby('date').agg({
                'measured_value': ['mean', 'std', 'count']
            }).reset_index()
            
            # 绘制实测值时间序列图(带误差线)
            fig = plt.figure(figsize=(15, 6))
            plt.errorbar(daily_measured['date'],
                        daily_measured['measured_value']['mean'],
                        yerr=daily_measured['measured_value']['std'],
                        fmt='o-',
                        capsize=5)
            plt.xlabel('date')
            plt.ylabel('measured_value')
            plt.title('measured_value_over_time(daily_statistics)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            
            # 将图形添加到PDF文档
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)
            
            # 绘制实测值散点图(每个样本)
            fig = plt.figure(figsize=(15, 6))
            plt.plot(measured_data['date'], measured_data['measured_value'], 
                    'o-', alpha=0.5, markersize=5)
            plt.xlabel('date')
            plt.ylabel('measured_value')
            plt.title('measured_value_over_time(each_sample)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)


            # 绘制按顺序的实测值变化图
            fig = plt.figure(figsize=(15, 6))
            plt.plot(measured_data['measured_value'], 'b-')
            # 在每个日期变化点添加竖线
            date_changes = np.where(measured_data['date'].diff() != pd.Timedelta(0))[0]
            for idx in date_changes:
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('sample_index')
            plt.ylabel('measured_value')
            plt.title('measured_value_over_time(by_collection_order)')
            plt.grid(True)
            plt.tight_layout()
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)



            # 绘制实测值与光谱强度的顺序变化图（分别归一化后画在一起）
            fig = plt.figure(figsize=(15, 6))
            # 归一化实测值
            normalized_measured = (measured_data['measured_value'] - measured_data['measured_value'].min()) / (measured_data['measured_value'].max() - measured_data['measured_value'].min())
            # 归一化光谱强度
            mean_intensity = np.mean(self.dataset['spectra'], axis=1)
            normalized_intensity = (mean_intensity - mean_intensity.min()) / (mean_intensity.max() - mean_intensity.min())
            # 绘制两条线
            plt.plot(normalized_measured, 'b-', label='normalized_measured_value')
            plt.plot(normalized_intensity, 'r-', label='normalized_spectral_intensity')
            plt.xlabel('sample_index')
            plt.ylabel('normalized_value')
            plt.title('normalized_comparison_between_measured_value_and_spectral_intensity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)


            # 按日期分组绘制光谱图
            unique_dates = pd.to_datetime(measured_data['date']).dt.date.unique()
            for date in unique_dates:
                # 获取当天的数据
                date_mask = pd.to_datetime(measured_data['date']).dt.date == date
                date_measured = measured_data[date_mask]
                
                # 绘制当天的实测值变化图
                fig = plt.figure(figsize=(15, 6))
                plt.plot(range(len(date_measured)), date_measured['measured_value'], 
                        'o-', alpha=0.5, markersize=5)
                plt.xlabel('sample_index')
                plt.ylabel('measured_value')
                plt.title(f'{date} measured_value_over_time')
                plt.grid(True)
                plt.tight_layout()
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # 绘制当天的光谱强度与实测值对比图
                fig = plt.figure(figsize=(15, 6))
                # 归一化当天的实测值
                norm_measured = (date_measured['measured_value'] - date_measured['measured_value'].min()) / \
                            (date_measured['measured_value'].max() - date_measured['measured_value'].min())
                # 归一化当天的光谱强度
                day_intensity = np.mean(self.dataset['spectra'][date_mask], axis=1)
                norm_intensity = (day_intensity - day_intensity.min()) / \
                            (day_intensity.max() - day_intensity.min())
                
                plt.plot(norm_measured, 'b-', label='normalized_measured_value')
                plt.plot(norm_intensity, 'r-', label='normalized_spectral_intensity')
                plt.xlabel('sample_index')
                plt.ylabel('normalized_value')
                plt.title(f'{date} normalized_comparison_between_measured_value_and_spectral_intensity')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)


        
        return daily_stats
    
    def _analyze_volunteer_patterns(self):
        """analyze_volunteer_patterns"""
        if 'volunteer' not in self.dataset:
            raise ValueError("the_dataset_is_missing_the_information_of_volunteer")
            
        # 计算每个志愿者的平均光谱
        volunteer_means = pd.DataFrame({
            'volunteer': self.dataset['volunteer'],
            'mean_intensity': np.mean(self.dataset['spectra'], axis=1)
        })
        
        # 创建志愿者统计信息
        volunteer_stats = volunteer_means.groupby('volunteer').agg({
            'mean_intensity': ['mean', 'std', 'count']
        }).reset_index()
        
        # 绘制志愿者箱线图
        fig = plt.figure(figsize=(15, 6))
        sns.boxplot(data=volunteer_means, x='volunteer', y='mean_intensity')
        plt.xlabel('volunteer_id')
        plt.ylabel('mean_spectral_intensity')
        plt.title('mean_spectral_intensity_distribution_of_each_volunteer')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        return volunteer_stats
    
    def _analyze_noise_levels(self):
        """analyze_noise_levels"""
        if 'spectra' not in self.dataset:
            raise ValueError("the_dataset_is_missing_the_information_of_spectra")
            
        # 将光谱数据按照每3个一组进行分组
        n_groups = len(self.spectral_data) // 3
        if n_groups == 0:
            self.add_paragraph("warning:the_number_of_samples_is_less_than_3,cannot_analyze_noise_levels")
            return
            
        X_grouped = np.array(self.spectral_data[:3*n_groups]).reshape(n_groups, 3, -1)

        # 计算每组在每个波长点上的标准差
        noise_levels = np.std(X_grouped, axis=1)  # 形状为 (n_groups, n_features)

        # 计算所有组的平均噪声水平
        mean_noise_levels = np.mean(noise_levels, axis=0)  # 形状为 (n_features,)

        # 绘制噪声水平图
        fig = plt.figure(figsize=(15, 6))
        plt.plot(mean_noise_levels, 'b-', label='mean_noise_level')
        plt.xlabel('wavelength_index')
        plt.ylabel('noise_level(std)')
        plt.title('mean_noise_level_of_each_wavelength_index')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 添加统计信息
        self.add_paragraph("noise_level_analysis_result:")
        self.add_paragraph(f"mean_noise_level: {np.mean(mean_noise_levels):.4f}")
        self.add_paragraph(f"max_noise_level: {np.max(mean_noise_levels):.4f}")
        self.add_paragraph(f"min_noise_level: {np.min(mean_noise_levels):.4f}")
        
        # 找出噪声最大的波长点
        max_noise_idx = np.argmax(mean_noise_levels)
        self.add_paragraph(f"the_index_of_the_wavelength_point_with_the_highest_noise: {max_noise_idx}")
        
        return mean_noise_levels

def generate_analysis_report(dataset, output_path='data_analysis_report.pdf'):
    """
    生成数据分析报告的便捷函数
    
    Parameters:
    -----------
    dataset : dict
        the_dataset_dictionary
    output_path : str, optional
        the_path_of_the_output_PDF_file
        
    Returns:
    --------
    str or None
        the_path_of_the_report_if_success,None_if_failed
    """
    try:
        analyzer = SpectralAnalysisReport(dataset, output_path)
        analyzer.analyze_and_generate_report()
        return output_path
    except Exception as e:
        print(f"error_occurred_when_generating_the_report: {str(e)}")
        print(f"error_occurred_at: {e.__traceback__.tb_frame.f_code.co_filename} 第 {e.__traceback__.tb_lineno} 行")
        return None

# 使用示例
if __name__ == "__main__":
    now_time = datetime.datetime.now()
    try:
        # 加载数据
        from nirapi.load_data import *
        
        # 从数据库获取数据集
        dataset_X = get_dataset_from_mysql(database='光谱数据库',table_name="复享光谱仪", project_name="多发光单收光探头血糖数据", X_type=['光谱',"采集日期","志愿者"])
        """
        dataset_X = {
            "光谱": np.array of shape (n_samples,n_feats)   # 这个是必须要有
            "实测值": np.array of shape (n_samples,)        # 这个是必须要有
            "其他": np.array of shape (n_samples,n_feats)     # 这个是可以有,但是不是必须要有
        } 
         
        """
        # 生成报告
        output_path = '数据分析报告.pdf'
        report_path = generate_analysis_report(dataset_X, output_path)
        
        if report_path:
            print(f"the_report_has_been_successfully_generated: {report_path}")
            
            # 尝试自动打开生成的PDF文件
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(report_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open {report_path}')
                else:  # Linux
                    os.system(f'xdg-open {report_path}')
            except Exception as e:
                print(f"cannot_automatically_open_the_PDF_file: {str(e)}")
                print(f"please_open_the_file_manually: {report_path}")
        else:
            print("the_report_generation_failed")
            
    except Exception as e:
        print(f"error_occurred_during_the_program_execution: {str(e)}")
        
    finally:
        # 清理matplotlib图形
        plt.close('all')
        print(f"the_execution_time_of_the_program: {(datetime.datetime.now() - now_time).total_seconds() / 60:.2f}minutes")