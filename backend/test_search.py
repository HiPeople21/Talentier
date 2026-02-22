"""Quick manual test for LinkedIn keyword profile search."""

from linkedin_search import search_linkedin_profiles


def main() -> None:
    profiles = search_linkedin_profiles(
        keywords=["python", "backend engineer"],
        location="United States",
        page=1,
        page_size=5,
    )

    print(f"Found {len(profiles)} profile(s)")
    for idx, p in enumerate(profiles, 1):
        print(f"[{idx}] {p['name']}")
        print(f"    {p['profile_url']}")
        print(f"    {p['headline']}")
        print()


if __name__ == "__main__":
    main()
