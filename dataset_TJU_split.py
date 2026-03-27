
material = 'NCM'
condition = 'CY25-05_1'
#NCA: 'CY45-05_1'  'CY25-05_1'  'CY25-025_1'  'CY25-1_1'  'CY35-05_1'
#NCM: 'CY25-05_1'  'CY35-05_1'  'CY45-05_1'
if material == 'NCA':
    if condition == 'CY45-05_1':
        train_files = [
             'CY45-05_1-#2.csv', 'CY45-05_1-#26.csv','CY45-05_1-#22.csv','CY45-05_1-#24.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv',  'CY45-05_1-#10.csv',
            'CY45-05_1-#12.csv','CY45-05_1-#17.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv',
            'CY45-05_1-#21.csv','CY45-05_1-#27.csv'
        ]
        val_files = [
            'CY45-05_1-#8.csv','CY45-05_1-#3.csv','CY45-05_1-#1.csv','CY45-05_1-#19.csv', 'CY45-05_1-#25.csv', 'CY45-05_1-#9.csv'
        ]
        test_files = [
            'CY45-05_1-#18.csv',  'CY45-05_1-#20.csv','CY45-05_1-#23.csv',
             'CY45-05_1-#28.csv', 'CY45-05_1-#4.csv','CY45-05_1-#11.csv'
        ]
    elif condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv', 'CY25-05_1-#18.csv',
            'CY25-05_1-#5.csv', 'CY25-05_1-#14.csv', 'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv',
            'CY25-05_1-#9.csv',  'CY25-05_1-#10.csv','CY25-05_1-#16.csv'
        ]
        val_files = [
            'CY25-05_1-#19.csv' ,'CY25-05_1-#15.csv','CY25-05_1-#13.csv', 'CY25-05_1-#12.csv'
        ]
        test_files = [
            'CY25-05_1-#11.csv','CY25-05_1-#1.csv', 'CY25-05_1-#8.csv', 'CY25-05_1-#17.csv'
        ]
    elif condition == 'CY25-025_1':
        train_files = [
            'CY25-025_1-#1.csv', 'CY25-025_1-#2.csv', 'CY25-025_1-#3.csv', 'CY25-025_1-#6.csv'
        ]
        val_files = [
            'CY25-025_1-#4.csv'
        ]
        test_files = [
            'CY25-025_1-#5.csv'
        ]
    elif condition == 'CY25-1_1':
        train_files = [
           'CY25-1_1-#1.csv','CY25-1_1-#5.csv','CY25-1_1-#7.csv','CY25-1_1-#6.csv', 'CY25-1_1-#4.csv', 'CY25-1_1-#9.csv'

        ]
        val_files = [
            'CY25-1_1-#3.csv'
        ]
        test_files = [
             'CY25-1_1-#2.csv','CY25-1_1-#8.csv'
        ]
    elif condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#2.csv'
        ]
        val_files = [
            'CY35-05_1-#3.csv'
        ]
        test_files = [
            'CY35-05_1-#1.csv'
        ]


elif material=='NCM':
    if condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#15.csv', 'CY25-05_1-#21.csv','CY25-05_1-#2.csv', 'CY25-05_1-#3.csv','CY25-05_1-#19.csv',
            'CY25-05_1-#6.csv', 'CY25-05_1-#8.csv','CY25-05_1-#22.csv', 'CY25-05_1-#16.csv',
             'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#18.csv',
        ]
        val_files = [
            'CY25-05_1-#7.csv', 'CY25-05_1-#14.csv','CY25-05_1-#17.csv', 'CY25-05_1-#23.csv'
        ]
        test_files = [
            'CY25-05_1-#1.csv', 'CY25-05_1-#13.csv',  'CY25-05_1-#20.csv','CY25-05_1-#4.csv'
        ]

    elif condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#25.csv','CY45-05_1-#17.csv','CY45-05_1-#7.csv',
            'CY45-05_1-#13.csv','CY45-05_1-#14.csv','CY45-05_1-#24.csv','CY45-05_1-#23.csv',
            'CY45-05_1-#9.csv',   'CY45-05_1-#12.csv','CY45-05_1-#3.csv', 'CY45-05_1-#20.csv',
            'CY45-05_1-#21.csv','CY45-05_1-#19.csv','CY45-05_1-#8.csv',
            'CY45-05_1-#1.csv','CY45-05_1-#26.csv','CY45-05_1-#4.csv'
        ]
        val_files = [
             'CY45-05_1-#22.csv','CY45-05_1-#15.csv',
            'CY45-05_1-#27.csv','CY45-05_1-#6.csv','CY45-05_1-#10.csv'
        ]
        test_files = [
             'CY45-05_1-#28.csv','CY45-05_1-#11.csv','CY45-05_1-#16.csv','CY45-05_1-#18.csv',
            'CY45-05_1-#2.csv', 'CY45-05_1-#5.csv'
        ]

    elif condition == 'CY35-05_1':
        train_files = [
             'CY35-05_1-#1.csv', 'CY35-05_1-#4.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv'
        ]