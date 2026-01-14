"""
Utility script to update and validate country mappings.

This script helps maintain the country_mappings.csv file by:
- Validating the CSV structure
- Checking for duplicates
- Adding new countries
- Updating alternative names

Usage:
    python utils/update_country_mappings.py --validate
    python utils/update_country_mappings.py --add "New Country" "ABC"
    python utils/update_country_mappings.py --add-alt "Germany" "GER"
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def load_mappings(csv_path='data/country_mappings.csv'):
    """Load the country mappings CSV file."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        sys.exit(1)
    return pd.read_csv(csv_path)


def save_mappings(df, csv_path='data/country_mappings.csv'):
    """Save the country mappings CSV file."""
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")


def validate_mappings(df):
    """Validate the country mappings for common issues."""
    issues = []
    
    # Check for required columns
    required_cols = ['country', 'abbreviation', 'alternative_names']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for duplicate countries
    duplicates = df[df.duplicated(subset=['country'], keep=False)]
    if not duplicates.empty:
        issues.append(f"Duplicate countries: {duplicates['country'].tolist()}")
    
    # Check for duplicate abbreviations (excluding alternatives)
    dup_abbrev = df[df.duplicated(subset=['abbreviation'], keep=False)]
    if not dup_abbrev.empty:
        issues.append(f"Duplicate abbreviations: {dup_abbrev['abbreviation'].tolist()}")
    
    # Check for empty country names
    empty_countries = df[df['country'].isna() | (df['country'] == '')]
    if not empty_countries.empty:
        issues.append(f"Empty country names at rows: {empty_countries.index.tolist()}")
    
    # Check for empty abbreviations
    empty_abbrev = df[df['abbreviation'].isna() | (df['abbreviation'] == '')]
    if not empty_abbrev.empty:
        issues.append(f"Empty abbreviations at rows: {empty_abbrev.index.tolist()}")
    
    # Check abbreviation length (should be 3 characters for ISO codes)
    invalid_length = df[df['abbreviation'].str.len() != 3]
    if not invalid_length.empty:
        issues.append(f"Non-standard abbreviation length: {invalid_length[['country', 'abbreviation']].to_dict('records')}")
    
    return issues


def add_country(df, country_name, abbreviation, alternative_names=''):
    """Add a new country to the mappings."""
    # Check if country already exists
    if country_name in df['country'].values:
        print(f"Warning: {country_name} already exists!")
        return df
    
    # Check if abbreviation already exists
    if abbreviation in df['abbreviation'].values:
        print(f"Warning: Abbreviation {abbreviation} already used!")
        return df
    
    # Add new row
    new_row = pd.DataFrame({
        'country': [country_name],
        'abbreviation': [abbreviation],
        'alternative_names': [alternative_names]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values('country').reset_index(drop=True)
    
    print(f"✓ Added: {country_name} ({abbreviation})")
    return df


def add_alternative(df, country_name, alternative):
    """Add an alternative name/abbreviation to an existing country."""
    if country_name not in df['country'].values:
        print(f"Error: {country_name} not found!")
        return df
    
    idx = df[df['country'] == country_name].index[0]
    current_alts = df.at[idx, 'alternative_names']
    
    if pd.isna(current_alts) or current_alts == '':
        df.at[idx, 'alternative_names'] = alternative
    else:
        # Check if alternative already exists
        alts_list = current_alts.split('|')
        if alternative in alts_list:
            print(f"Warning: {alternative} already exists for {country_name}")
            return df
        df.at[idx, 'alternative_names'] = current_alts + '|' + alternative
    
    print(f"✓ Added alternative '{alternative}' to {country_name}")
    return df


def list_countries(df):
    """List all countries with their abbreviations."""
    print("\nCountry Mappings:")
    print("=" * 80)
    for _, row in df.iterrows():
        alts = row['alternative_names'] if pd.notna(row['alternative_names']) else ''
        alt_str = f" (also: {alts})" if alts else ''
        print(f"{row['country']:40} {row['abbreviation']:5} {alt_str}")
    print(f"\nTotal: {len(df)} countries")


def main():
    parser = argparse.ArgumentParser(description='Manage country mappings')
    parser.add_argument('--validate', action='store_true', help='Validate the mappings')
    parser.add_argument('--list', action='store_true', help='List all countries')
    parser.add_argument('--add', nargs=2, metavar=('COUNTRY', 'ABBREV'), 
                       help='Add a new country')
    parser.add_argument('--add-alt', nargs=2, metavar=('COUNTRY', 'ALTERNATIVE'),
                       help='Add alternative name to existing country')
    parser.add_argument('--csv', default='data/country_mappings.csv',
                       help='Path to CSV file (default: data/country_mappings.csv)')
    
    args = parser.parse_args()
    
    # Load mappings
    df = load_mappings(args.csv)
    modified = False
    
    # Validate
    if args.validate or not any([args.list, args.add, args.add_alt]):
        print("Validating country mappings...")
        issues = validate_mappings(df)
        if issues:
            print("\n❌ Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All validations passed!")
    
    # List
    if args.list:
        list_countries(df)
    
    # Add country
    if args.add:
        df = add_country(df, args.add[0], args.add[1])
        modified = True
    
    # Add alternative
    if args.add_alt:
        df = add_alternative(df, args.add_alt[0], args.add_alt[1])
        modified = True
    
    # Save if modified
    if modified:
        save_mappings(df, args.csv)


if __name__ == '__main__':
    main()
