
function csv_to_results(filepath)
    csv_reader = CSV.File(filepath)

    results = Results()
    for row in csv_reader
        logresult!(results, row.iterations, row.times, row.fvaluegaps, row.metricLPs)
    end
    return results
end
