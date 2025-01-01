def compute_weekly_min_max(self) -> pd.DataFrame:
    """
    1) For each (Year, WeekOfYear), find the min & max closing price.
    2) Compute each as a % difference from that year's opening price.
    3) Average those % differences across ALL years for each WeekOfYear (1..52).
    
    Returns a DataFrame with columns:
      ['WeekOfYear', 'AvgMinPercent', 'AvgMaxPercent']
    """
    df = self.df.copy()
    
    # -- STEP 1: Add columns for year & week-of-year --
    df['Year'] = df.index.year
    df['WeekOfYear'] = df.index.isocalendar().week  # 1..53
    # Some instruments might have 53rd week in certain years, handle if you like:
    df['WeekOfYear'] = df['WeekOfYear'].apply(lambda w: 52 if w > 52 else w)
    
    # -- STEP 2: Identify each year's opening (first trading day) price --
    # Create an integer day-of-year to find earliest trading day per year
    df['DayOfYear'] = df.index.dayofyear
    # We pick out the row(s) that match the earliest day for each year
    idx_earliest = df.groupby('Year')['DayOfYear'].transform('min') == df['DayOfYear']
    # Build a dict: {year: close_on_first_trading_day}
    year_open_dict = df[idx_earliest].set_index('Year')['close'].to_dict()
    
    # -- STEP 3: Group by (Year, WeekOfYear) to get weekly min & max close --
    grouped = df.groupby(['Year', 'WeekOfYear'])['close'].agg(['min','max']).reset_index()
    grouped.rename(columns={'min': 'MinPrice', 'max': 'MaxPrice'}, inplace=True)
    
    # Attach each group's year_open price
    grouped['YearOpen'] = grouped['Year'].map(year_open_dict)
    
    # -- STEP 4: Compute Min% and Max% vs. that year's opening price --
    grouped['MinPercent'] = ((grouped['MinPrice'] - grouped['YearOpen']) / grouped['YearOpen']) * 100
    grouped['MaxPercent'] = ((grouped['MaxPrice'] - grouped['YearOpen']) / grouped['YearOpen']) * 100
    
    # -- STEP 5: Now average MinPercent & MaxPercent by WeekOfYear across ALL years --
    result = grouped.groupby('WeekOfYear')[['MinPercent', 'MaxPercent']].mean().reset_index()
    
    # Rename columns to reflect that these are averaged values across years
    result.rename(columns={
        'MinPercent': 'AvgMinPercent',
        'MaxPercent': 'AvgMaxPercent'
    }, inplace=True)
    
    # Final columns: ['WeekOfYear', 'AvgMinPercent', 'AvgMaxPercent']
    return result
