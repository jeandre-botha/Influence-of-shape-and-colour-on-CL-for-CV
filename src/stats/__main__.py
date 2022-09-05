import argparse
import os
import glob
from logger import logger
import warnings
warnings.filterwarnings('ignore')
import xlsxwriter
from utils import add_headings, resolve_cell, set_column_autowidth


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='''Utility for summarizing results''')
        parser.add_argument(
            '-f',
            '--folder',
            type=str,
            nargs='?',
            required=True,
            help='Results root folder',
        )

        parser.add_argument(
            '-r',
            '--regex',
            type=str,
            nargs='?',
            required=False,
            default="",
            help='Regex for file match',
        )

        options = parser.parse_args()

        workbook = xlsxwriter.Workbook(os.path.join(options.folder, 'results.xlsx'))
        heading_format = workbook.add_format({'bold': True})
        worksheet = workbook.add_worksheet('Summary')

        row = 0
        add_headings(worksheet, heading_format)
        row += 1

        column_index = {
            'run': 0,
            'loss': 1,
            'average': 2
        }

        for file_path in glob.iglob('{}/**'.format(options.folder), recursive=True):
            if os.path.isfile(file_path):
                file_name = os.path.split(file_path)[-1]
                file_extension = os.path.splitext(file_name)[1].lower()
                if file_extension=='.txt':
                    with open(file_path, "r") as f:
                        file_content = f.read()
                        loss_str, acc_str = file_content.split(',')
                        loss = loss_str.split(':')[1].strip()
                        acc = acc_str.split(':')[1].strip()
                        worksheet.write_number(resolve_cell(row, 0), int(row))
                        worksheet.write_number(resolve_cell(row, 1), float(loss))
                        worksheet.write_number(resolve_cell(row, 2), float(acc))
                        row += 1

        worksheet.write(resolve_cell(row, column_index['run']), 'Average', heading_format)
        worksheet.write_formula(
            resolve_cell(row, column_index['loss']),
            '=AVERAGE({}:{})'.format(resolve_cell(1, column_index['loss']), resolve_cell(row-1, column_index['loss'])))
        worksheet.write_formula(
            resolve_cell(row, column_index['average']),
            '=AVERAGE({}:{})'.format(resolve_cell(1, 2), resolve_cell(row-1, column_index['average'])))
        worksheet.write(resolve_cell(row+1, column_index['run']), 'Standard deviation', heading_format)
        worksheet.write_formula(
            resolve_cell(row+1, column_index['loss']),
            '=_xlfn.STDEV.S({}:{})'.format(resolve_cell(1, column_index['loss']), resolve_cell(row-1, column_index['loss'])))
        worksheet.write_formula(
            resolve_cell(row+1, column_index['average']),
            '=_xlfn.STDEV.S({}:{})'.format(resolve_cell(1, 2), resolve_cell(row-1, column_index['average'])))
        
        # set column width
        for _, index in column_index.items():
            set_column_autowidth(worksheet, column= index)
        
        workbook.close()

    except Exception as ex:
        logger.exception(ex)
