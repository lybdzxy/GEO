import cdsapi
import time
import requests
import multiprocessing

c = cdsapi.Client()#timeout=300
years = [ str(id1) for id1 in range(1950,1960) ]
months =[ '%02d' % id2 for id2 in range(1,13) ]
var = "Total precipitation"

def Download(iyear, imonth):
    t000=time.time()
    result = c.retrieve(
        'reanalysis-era5-land',
        {
                'variable': 'total_precipitation',
                "year": iyear,
                "month": imonth,
            'day': [
                '01', '02', '03', '04',
                '05', '06', '07',
                '08', '09', '10',
                '11', '12', '13',
                '14', '15', '16',
                '17', '18', '19',
                '20', '21', '22',
                '23', '24', '25',
                '26', '27', '28',
                '29', '30', '31',
            ],
            'time': '00:00',
                #
                # Users can change the output grid resolution and selected area
                #
                #                "grid": "1.0/1.0",
                #                "area":{"lat": [10, 60], "lon": [65, 140]}

        })

    # set name of output file for each month (statistic, variable, year, month

    file_name =  'ERA5' + "_" + var +  iyear + imonth + ".nc"

    location = result[0]['location']
    res = requests.get(location, stream=True)
    print("Writing data to " + file_name)
    with open(file_name, 'wb') as fh:
        for r in res.iter_content(chunk_size=1024):
            fh.write(r)
    fh.close()
    print('***样本%s 保存结束, 耗时: %.3f s / %.3f mins****************' % (file_name,(time.time() - t000), (time.time() - t000) / 60))


if __name__ == "__main__":
    t0 = time.time()
    # ===================================================================================
    print('*****************程序开始*********************')
    for yr in years:
        for mn in months:
            Download(yr, mn)
    print('***********************程序结束, 耗时: %.3f s / %.3f mins****************' % (
        (time.time() - t0), (time.time() - t0) / 60))