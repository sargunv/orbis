//! `#[derive(Wire)]`: a struct's fields in declaration order, an enum's
//! variant index (a varint) then its fields. A field marked `#[wire(skip)]`
//! isn't sent and arrives as its `Default`. Each type parameter must be
//! `Wire` too.

#![forbid(unsafe_code)]

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Index, parse_macro_input, parse_quote};

#[proc_macro_derive(Wire, attributes(wire))]
pub fn derive_wire(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);
    for param in input.generics.type_params_mut() {
        param.bounds.push(parse_quote!(::orb_wire::Wire));
    }
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let (put, take) = match &input.data {
        Data::Struct(data) => {
            let (pattern, puts, take) = fields(&data.fields, quote!(Self));
            (
                quote! { let #pattern = self; #puts },
                quote! { Some(#take) },
            )
        }
        Data::Enum(data) => {
            let mut puts = Vec::new();
            let mut takes = Vec::new();
            for (index, variant) in data.variants.iter().enumerate() {
                let index = u32::try_from(index).expect("fewer than 2^32 variants");
                let ident = &variant.ident;
                let (pattern, fields_put, take) = fields(&variant.fields, quote!(Self::#ident));
                puts.push(quote! {
                    #pattern => {
                        ::orb_wire::Wire::put(&#index, out);
                        #fields_put
                    }
                });
                takes.push(quote! { #index => Some(#take), });
            }
            (
                quote! { match self { #(#puts)* } },
                quote! {
                    match <u32 as ::orb_wire::Wire>::take(input)? {
                        #(#takes)*
                        _ => None,
                    }
                },
            )
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(&input.ident, "Wire can't be derived for a union")
                .to_compile_error()
                .into();
        }
    };
    quote! {
        impl #impl_generics ::orb_wire::Wire for #name #ty_generics #where_clause {
            fn put(&self, out: &mut ::std::vec::Vec<u8>) {
                #put
            }
            fn take(input: &mut &[u8]) -> ::core::option::Option<Self> {
                use ::core::option::Option::{None, Some};
                #take
            }
        }
    }
    .into()
}

/// For `fields` of `path` (a struct, or an enum's variant): the pattern
/// that binds them, the statements that put them, and the expression that
/// takes them, in declaration order.
fn fields(fields: &Fields, path: TokenStream) -> (TokenStream, TokenStream, TokenStream) {
    let skipped = |field: &syn::Field| -> syn::Result<bool> {
        let mut skip = false;
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("wire")) {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip") {
                    skip = true;
                    Ok(())
                } else {
                    Err(meta.error("the one `wire` attribute is `skip`"))
                }
            })?;
        }
        Ok(skip)
    };
    let mut bindings = Vec::new();
    let mut puts = Vec::new();
    let mut takes = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        // A misspelt attribute fails the build rather than sending the field.
        let skip = skipped(field).unwrap_or_else(|e| {
            puts.push(e.to_compile_error());
            false
        });
        let binding = format_ident!("field{i}");
        let take = if skip {
            quote!(::core::default::Default::default())
        } else {
            puts.push(quote!(::orb_wire::Wire::put(#binding, out);));
            quote!(::orb_wire::Wire::take(input)?)
        };
        let bound = if skip { quote!(_) } else { quote!(#binding) };
        match &field.ident {
            Some(ident) => {
                bindings.push(quote!(#ident: #bound));
                takes.push(quote!(#ident: #take));
            }
            None => {
                let index = Index::from(i);
                bindings.push(quote!(#index: #bound));
                takes.push(quote!(#index: #take));
            }
        }
    }
    (
        quote!(#path { #(#bindings),* }),
        quote!(#(#puts)*),
        quote!(#path { #(#takes),* }),
    )
}
