create extension if not exists pgcrypto;

create table if not exists public.team_members (
    id uuid primary key default gen_random_uuid(),
    telegram_user_id bigint not null unique,
    telegram_username text,
    display_name text not null,
    role text not null default 'member' check (role in ('admin','member')),
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists public.creators (
    id uuid primary key default gen_random_uuid(),
    email text not null,
    normalized_email text not null unique,
    instagram_username text,
    normalized_username text,
    full_name text,
    followers bigint,
    following bigint,
    post_count bigint,
    location text,
    niche text,
    bio text,
    first_added_by uuid references public.team_members(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists creators_username_unique
on public.creators(normalized_username)
where normalized_username is not null;

create index if not exists creators_location_idx on public.creators(location);
create index if not exists creators_niche_idx on public.creators(niche);
create index if not exists creators_created_at_idx on public.creators(created_at);

create table if not exists public.creator_submissions (
    id uuid primary key default gen_random_uuid(),
    team_member_id uuid not null references public.team_members(id),
    creator_id uuid references public.creators(id),
    telegram_message_id bigint,
    status text not null,
    extracted_username text,
    extracted_email text,
    extracted_data jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists creator_submissions_member_date_idx
on public.creator_submissions(team_member_id, created_at);

-- Atomic insert: only the first valid normalized email earns a unique-lead credit.
create or replace function public.register_creator_lead(
    p_email text,
    p_username text default null,
    p_full_name text default null,
    p_followers bigint default null,
    p_following bigint default null,
    p_post_count bigint default null,
    p_location text default null,
    p_niche text default null,
    p_bio text default null,
    p_member_id uuid default null
)
returns table(creator_id uuid, is_new boolean)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_id uuid;
    v_new boolean := false;
    v_email text;
    v_username text;
begin
    v_email := lower(trim(p_email));
    v_username := nullif(lower(trim(p_username)), '');

    if v_email is null or v_email = '' or v_email !~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$' then
        raise exception 'A valid email is required';
    end if;

    insert into public.creators (
        email, normalized_email, instagram_username, normalized_username,
        full_name, followers, following, post_count, location, niche, bio, first_added_by
    ) values (
        trim(p_email), v_email, p_username, v_username,
        p_full_name, p_followers, p_following, p_post_count, p_location, p_niche, p_bio, p_member_id
    )
    on conflict (normalized_email) do nothing
    returning id into v_id;

    if v_id is not null then
        v_new := true;
    else
        select c.id into v_id
        from public.creators c
        where c.normalized_email = v_email
        for update;

        update public.creators c set
            instagram_username = coalesce(c.instagram_username, p_username),
            normalized_username = coalesce(c.normalized_username, v_username),
            full_name = coalesce(c.full_name, p_full_name),
            followers = coalesce(c.followers, p_followers),
            following = coalesce(c.following, p_following),
            post_count = coalesce(c.post_count, p_post_count),
            location = coalesce(c.location, p_location),
            niche = coalesce(c.niche, p_niche),
            bio = coalesce(c.bio, p_bio),
            updated_at = now()
        where c.id = v_id;
    end if;

    return query select v_id, v_new;
end;
$$;

create or replace function public.get_member_daily_progress(
    p_member_id uuid,
    p_day date
)
returns table(email_count bigint)
language sql
security definer
set search_path = public
as $$
    select count(*)::bigint
    from public.creators c
    where c.first_added_by = p_member_id
      and (c.created_at at time zone 'Asia/Kolkata')::date = p_day;
$$;

-- Bot uses the Supabase service-role key server-side. Keep it out of frontend/client code.
-- RLS is enabled as a safe default for any future direct client access.
alter table public.team_members enable row level security;
alter table public.creators enable row level security;
alter table public.creator_submissions enable row level security;
